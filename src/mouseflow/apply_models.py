from abc import ABC, abstractmethod
import os
import numpy as np
import gdown
import yaml
from contextlib import nullcontext
import torch
from pathlib import Path
from torch.cuda.amp import autocast

import cv2
import pandas as pd
from ruamel.yaml import YAML


def download_models(models_dir, facemodel_name, bodymodel_name):   
    if not os.path.exists(os.path.join(models_dir, facemodel_name)):
        dlc_face_url = 'https://drive.google.com/drive/folders/1_XPPyzaxMjQ901vJCwtv1g_h5DYWHM8j?usp=sharing'
        gdown.download_folder(dlc_face_url, models_dir, quiet=True, use_cookies=False)

    if not os.path.exists(os.path.join(models_dir, bodymodel_name)):
        dlc_body_url = 'https://drive.google.com/drive/folders/1_XPPyzaxMjQ901vJCwtv1g_h5DYWHM8j?usp=sharing'
        gdown.download_folder(dlc_body_url, models_dir, quiet=True, use_cookies=False)
    
    # PyTorch config (DLC 3)
    dlc_faceyaml = os.path.join(models_dir, facemodel_name, 'pytorch_config.yaml')
    dlc_bodyyaml = os.path.join(models_dir, bodymodel_name, 'pytorch_config.yaml')

    print(f"[download_models] → using facecfg = {dlc_faceyaml}")
    print(f"[download_models] → using bodycfg = {dlc_bodyyaml}")
    return dlc_faceyaml, dlc_bodyyaml

def detect_engine(cfg_path):
    """
    Return 'pytorch' if this config belongs to DLC 3,
    otherwise None.
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if (os.path.basename(cfg_path).startswith('pytorch_')   # DLC 3 puts a prefix
        or cfg.get('engine') == 'pytorch'):                 # or DLC 3 may embed key
        return 'pytorch'
    return None


class KeypointDetector(ABC):
    @abstractmethod
    def load_model_cfg(self, model_cfg_path):
        ...
    
    @abstractmethod
    def detect_keypoints(self,
        video_file: str,
        out_dir: str,
        make_labeled_video: bool = False,
        overwrite: bool = False):
        ...


class LPDetector(KeypointDetector):
    def __init__(self, model_cfg_path, model_path):
        self.cfg = self.load_model_cfg(model_cfg_path)
        # print(f"config loaded: {self.cfg}")
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_model_cfg(self, model_cfg_path):
        # self.model = load_model_from_checkpoint(model_path).eval().cuda()
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(model_cfg_path)
        cfg.model.backbone = cfg.model.get("backbone", "resnet50")
        cfg.model.pretrained = False
        return cfg

    
    def detect_keypoints(self,
        video_file: str,
        out_dir: str,
        make_labeled_video: bool = False,
        overwrite: bool = False):
        from lightning_pose.utils.predictions import export_predictions_and_labeled_video

        os.makedirs(out_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(video_file))[0]
        preds_csv = os.path.join(out_dir, f"{base}.csv")
        labeled_mp4 = os.path.join(out_dir, f"{base}_labeled.mp4") if make_labeled_video else None

        if not overwrite and os.path.isfile(preds_csv):
            return preds_csv, labeled_mp4
        autocast_enabled = autocast() if self.device == "cuda" else nullcontext()
        with torch.no_grad():
            with autocast_enabled:
                export_predictions_and_labeled_video(
                    video_file=video_file,
                    cfg=self.cfg,
                    prediction_csv_file=preds_csv,
                    ckpt_file=self.model_path,      # uses checkpoint path
                    labeled_mp4_file=labeled_mp4,  # None skips rendering
                )
        return preds_csv, labeled_mp4
    

class DLCDetector(KeypointDetector):
    def __init__(self, model_cfg_path, model_path, shuffle=3):
        self.model_path = model_path
        self.model_cfg = model_cfg_path
        self.shuffle = shuffle
        self.pose_cfg = Path(os.path.dirname(self.model_path)).parent / "test" /  "pose_cfg.yaml"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("inited!")

    def create_labeled_video(
        self,
        video_path: str,
        h5_path: str,
        pose_cfg_path: str,
        out_path: str | None = None,
        pcutoff: float = 0.1,
        radius: int = 3,
        thickness: int = 2,
        draw_names: bool = False
    ):
        """
        Minimal labeled video writer for DLC single-animal outputs.
        - video_path: path to the original video
        - h5_path: DLC predictions (.h5) with MultiIndex columns (scorer, bodypart, coord)
        - pose_cfg_path: test/pose_cfg.yaml containing metadata.bodyparts
        - out_path: output mp4 path (defaults to <video_basename>_labeled.mp4 next to input)
        - pcutoff: confidence threshold for drawing
        """
        # load bodyparts from pose config
        yaml = YAML(typ="safe")
        pose_cfg = yaml.load(open(pose_cfg_path, "r"))
        bodyparts = pose_cfg.get("metadata", {}).get("bodyparts", [])
        if not bodyparts:
            raise ValueError("No bodyparts found in pose_cfg.yaml under metadata.bodyparts.")

        # load predictions
        df = pd.read_hdf(h5_path)
        if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 3:
            raise ValueError("Unexpected H5 columns. Expect MultiIndex (scorer, bodypart, coord).")

        # map: bodypart -> (x_col, y_col, l_col)
        # pick the first scorer layer present
        scorers = sorted(set([c[0] for c in df.columns]))
        scorer = scorers[0]

        def col(bp, coord):
            # handle possible coord names: 'x','y','likelihood' or 'likelihoods'
            candidates = [(scorer, bp, coord)]
            if coord == "likelihood":
                candidates.append((scorer, bp, "likelihoods"))
            for c in candidates:
                if c in df.columns:
                    return c
            return None

        triplets = []
        for bp in bodyparts:
            xC = col(bp, "x")
            yC = col(bp, "y")
            lC = col(bp, "likelihood")
            if xC and yC and lC:
                triplets.append((bp, xC, yC, lC))
            else:
                # skip bodypart if any column missing
                continue

        if not triplets:
            raise ValueError("No valid (x,y,likelihood) columns found for listed bodyparts.")

        # open video IO
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        nframes_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # align lengths
        nframes_pred = len(df)
        nframes = min(nframes_vid, nframes_pred)

        if out_path is None:
            base, _ = os.path.splitext(video_path)
            out_path = base + "_labeled.mp4"

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            cap.release()
            raise IOError(f"Could not open writer: {out_path}")

        # simple deterministic colors per bodypart
        def bp_color(i, total):
            hue = int(179 * i / max(1, total))
            col = cv2.cvtColor(np.uint8([[[hue, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0, :]
            return int(col[0]), int(col[1]), int(col[2])

        colors = {bp: bp_color(i, len(triplets)) for i, (bp, *_cols) in enumerate(triplets)}

        # iterate frames
        for idx in range(nframes):
            ok, frame = cap.read()
            if not ok:
                break
            row = df.iloc[idx]
            for bp, xC, yC, lC in triplets:
                x = row[xC]
                y = row[yC]
                l = row[lC]
                if pd.isna(x) or pd.isna(y) or pd.isna(l) or float(l) < pcutoff:
                    continue
                cx, cy = int(round(float(x))), int(round(float(y)))
                cv2.circle(frame, (cx, cy), radius, colors[bp], thickness, lineType=cv2.LINE_AA)
                if draw_names:
                    cv2.putText(frame, bp, (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colors[bp], 1, cv2.LINE_AA)
            writer.write(frame)

        cap.release()
        writer.release()
        return out_path

    def load_model_cfg(self, model_cfg_path):
        return os.path.join(model_cfg_path, "config.yaml")
    
    def detect_keypoints(self,
        video_file, out_dir,
        make_labeled_video = False,
        overwrite = False):

        from deeplabcut.compat import analyze_videos
        from deeplabcut.core.engine import Engine

        os.makedirs(out_dir, exist_ok=True)
        autocast_enabled = autocast() if self.device == "cuda" else nullcontext()

        with torch.no_grad():
            with autocast_enabled:
                analyze_videos(
                    config=self.model_cfg,
                    videos=[video_file],
                    shuffle=self.shuffle,
                    videotype=None,        # e.g. "mp4" or None
                    destfolder=out_dir,
                    engine=Engine.PYTORCH,
                    modelprefix=os.path.dirname(self.model_cfg),    # directory of pytorch_config.yaml
                    device=str(self.device) if self.device == 'cuda' else None,
            )
        
            if make_labeled_video:
                vid_stem = Path(video_file).stem
                h5_out_candidates = sorted(
                    Path(out_dir).glob(f"{vid_stem}*shuffle{self.shuffle}*.h5"),
                    key=lambda p: p.stat().st_mtime,
                    reverse=True
                )
                if not h5_out_candidates:
                    raise FileNotFoundError(f"h5 dlc output file for video {vid_stem} not found")
                h5_out = str(h5_out_candidates[0])
                self.create_labeled_video(video_file,
                    h5_out,
                    str(self.pose_cfg))