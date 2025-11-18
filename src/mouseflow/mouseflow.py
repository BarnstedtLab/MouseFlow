#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# import gdown
import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.stats import zscore

from dataclasses import dataclass
from pathlib import Path

import mouseflow.body_processing as body_processing
import mouseflow.face_processing as face_processing
from mouseflow.utils import motion_processing, confidence_na, process_raw_data


@dataclass(frozen=True)
class MFConfig:
    dgp: bool = True
    conf_thresh: float | None = None
    interpolation_limits_sec: dict = ...
    smoothing_windows_sec: dict = ...
    na_limit: float = 0.25
    faceregions_sizes: dict | None = None
    base_resolution: tuple[int,int] | None = None
    manual_anchor: dict | None = None
    of_backend: str = "RAFT"
    overwrite: bool = False

class MouseFlow:
    def __init__(self, dlc_dir: str | Path, cfg: MFConfig):
        self.dlc_dir = Path(dlc_dir)
        self.cfg = cfg

    def run(self):
        files = self._analysis_files()
        face_files = files['face_files']
        body_files = files['body_files']

        if not face_files:
            print(f"No face files found to process in {self.dlc_dir}.")
        if not body_files:
            print(f"No body files found to process in {self.dlc_dir}.")
        
        for ff in face_files:
            self.analyse_face(ff)
        for bf in body_files:
            self.analyse_body(bf)

    def _load_markers(self, file: Path, key: str):
        # Reading in DLC/DGP file
        markers = pd.read_hdf(file, mode='r', key=key)
        markers.columns = markers.columns.droplevel(0)
        return markers
    
    def _face_to_h5(self, out_file: Path, face_masks, face_anchor: pd.DataFrame, face: pd.DataFrame):
        mf_out_file = str(out_file)
        with h5py.File(mf_out_file, "a") as out:
            if "facemasks" in out:
                del out["facemasks"]
            out.create_dataset("facemasks", data=face_masks)
        face_anchor.to_hdf(mf_out_file, key="face_anchor", mode="a")
        face.to_hdf(mf_out_file, key="face", mode="a")
        print(f"saved mouseflow face analysis to {mf_out_file}")

    def _interpolation_limits(self, fps: float, min_frames: int=1):
        out : dict[str, int] = {}
        for key, val in self.cfg.interpolation_limits_sec.items():
            if val is None or (isinstance(val, (int, float)) and val <= 0):
                out[key] = min_frames
            else:
                out[key] = int(max(min_frames, round(float(val) * float(fps))))
        return out
   
    def _get_video_for_marker(self, marker_file: Path):
        VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
        """
        Return the single video in dlc_dir matching the marker file.
        Assumes 0 or 1 matching video per marker.
        Match rule: video filename starts with the marker name portion before 'DLC' (case-insensitive).
        """

        if marker_file.suffix.lower() != ".h5":
            raise ValueError(f"Expected .h5 marker file, got: {marker_file}")

        base = marker_file.stem
        # Find 'DLC' case-insensitively without regex
        i = base.lower().find("dlc")
        prefix = base[:i] if i != -1 else base
        prefix = prefix.rstrip("_-. ")

        candidates = [
            p for p in self.dlc_dir.parent.iterdir()
            if (p.is_file()
                and p.suffix.lower() in VIDEO_EXTS
                and p.name.startswith(prefix)
                and "labeled" not in p.name
            )
        ]

        if not candidates:
            raise FileNotFoundError(f"No video starting with '{prefix}' in {self.dlc_dir}")

        if len(candidates) > 1:
            raise RuntimeError(
                f"Expected at most one video for '{marker_file.name}', found: {[c.name for c in candidates]}")

        return candidates[0]

    def _video_info(self, video_path: Path | str):
        """
        Open a video and return (fps, width, height, cap).
        Caller owns 'cap' and should release it when done.
        Raises OSError if the file cannot be opened or has invalid properties.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise OSError(f"Invalid FPS ({fps}) for: {video_path}")

        if width <= 0 or height <= 0:
            cap.release()
            raise OSError(f"Invalid frame size {width}x{height} for: {video_path}")

        return float(fps), width, height, cap

    def _analysis_files(self):
        '''
        Returns face and body files hich can be found in dlc_dir and should be analysed.
        The face files must match the pattern '*DLC*MouseFace*.h5'.
        The body files must match the pattern '*DLC*MouseBody*.h5'.
        '''
        if self.dlc_dir is None:
            raise RuntimeError("No DLC directory defined.")
        d = Path(self.dlc_dir)

        face_raw = sorted(d.glob('*DLC*MouseFace*.h5'))
        body_raw = sorted(d.glob('*DLC*MouseBody*.h5'))

        def needs_processing(p: Path) -> bool:
            if p.name.lower().endswith("_mouseflow.h5"):
                return False
            mf = p.with_name(p.stem + '_mouseflow.h5')
            if mf.exists() and not self.cfg.overwrite:
                print(f"skipping file {p.name} as it is already analysed!") 
            return self.cfg.overwrite or not mf.exists()

        face_files = [p for p in face_raw if needs_processing(p)]
        body_files = [p for p in body_raw if needs_processing(p)]

        return {'face_files': face_files, 'body_files': body_files}
    
    def _has_gpu_support(self):
        '''
        Tests if the installation has proper GPU support (and if a GPU is actually available).
        First check: OpenCV with CUDA support (needs to be compiled on machine!!)
        Second check: PyTorch with CUDA support
        
        if 'Farneback' is specified as of_backend and there is no opencv CUDA installation,
        this will still return False even if it runs on a GPU system!!
        '''
        import torch
        has_cv2_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0
        has_pytorch_gpu = torch.cuda.is_available()

        if not has_cv2_cuda and (self.cfg.of_backend == "Farneback"):
            print("No Opencv with GPU support detected, sorry. Ensure to have a working nvidia GPU." \
                "If you have, try switching mf_config.of_backend to 'RAFT' ")
            return False
        
        if not has_pytorch_gpu and (self.cfg.of_backend == "RAFT"):
            print("No Pytorch with GPU support detected, sorry. Ensure to have a working nvidia GPU." \
                "If you have, try switching mf_config.of_backend to 'Farneback' ")
            return False

        if has_cv2_cuda and (self.cfg.of_backend == "Farneback"):
            return True
                
        if has_pytorch_gpu and (self.cfg.of_backend == "RAFT"):
            return True
        return False # safe fallback

    
    # Actual analysis

    # Face analysis
    def analyse_face(self, face_file: Path):
        out_file = face_file.with_name(face_file.stem + '_mouseflow.h5')
        if out_file.exists() and not self.cfg.overwrite:
            print(f"{out_file} already analysed, skipping ahead...")
            return
        markers_face = self._load_markers(face_file, key=None)
        video_file = self._get_video_for_marker(face_file)
        fps, w, h, cap = self._video_info(video_file)
        markers_face = confidence_na(self.cfg.dgp, self.cfg.conf_thresh, markers_face)
        interpolation_limits = self._interpolation_limits(fps)

        markers_face.loc[:, ['pupil'+str(n+1) for n in range(6)]] = \
            markers_face.loc[:, ['pupil'+str(n+1) for n in range(6)]].interpolate(
                method='linear', limit=interpolation_limits['pupil'])

        # Extracting pupil and eyelid data
        pupil_raw = face_processing.pupilextraction(
            markers_face[['pupil'+str(n+1) for n in range(6)]].values)
        eyelid_dist_raw = pd.Series(motion_processing.dlc_pointdistance2(
            markers_face['eyelid1'], markers_face['eyelid2']), name='EyeLidDist')

        # Define and save face regions
        face_masks, face_anchor = face_processing.define_faceregions(
            markers_face, video_file, face_file, manual_anchor=self.cfg.manual_anchor,
            base_resolution=self.cfg.base_resolution, faceregions_sizes=self.cfg.faceregions_sizes
        )

        # Extract motion in face regions
        if not self._has_gpu_support():
            print("No CUDA support detected. Processing without optical flow...")
            face_motion = face_processing.facemotion_nocuda(
                video_file, face_masks)
            face_raw = pd.concat([pupil_raw, eyelid_dist_raw, face_motion], axis=1)
        else:
            face_motion = face_processing.facemotion(video_file, face_masks, backend=self.cfg.of_backend) # switch between RAFT and Farneback is handled internally
            whisk_freq = motion_processing.freq_analysis2(
                face_motion['OFang_Whiskerpad'], fps, rollwin=fps, min_periods=int(fps*.67))
            sniff_freq = motion_processing.freq_analysis2(
                face_motion['OFang_Nose'],       fps, rollwin=fps, min_periods=int(fps*.67))
            chewenv, chew = motion_processing.hilbert_peaks(
                face_motion['OFang_Mouth'],    fps)
            face_freq = pd.DataFrame(
                {'Whisking_freq': whisk_freq, 'Sniff_freq': sniff_freq, 'Chewing_Envelope': chewenv, 'Chew': chew})
            face_raw = pd.concat([pupil_raw, eyelid_dist_raw, face_motion, face_freq], axis=1)
        cap.release()
        face = process_raw_data(self.cfg.smoothing_windows_sec, self.cfg.na_limit, fps, interpolation_limits, face_raw)
        self._face_to_h5(out_file, face_masks, face_anchor, face)

    def as_series(self, x, name, idx):
        x = pd.Series(x, name=name)
        return x.reindex(idx)

    def analyse_body(self, body_file: Path):
        out_file = body_file.with_name(body_file.stem + '_mouseflow.h5')
        if out_file.exists() and not self.cfg.overwrite:
            print(f"{out_file} already analysed, skipping ahead...")
            return
        
        markers_body = self._load_markers(body_file, key=None)
        video_file = self._get_video_for_marker(body_file)
        fps, w, h, cap = self._video_info(video_file)
        # markers_body = confidence_na(self.cfg.dgp, self.cfg.conf_thresh, markers_body)
        interpolation_limits = self._interpolation_limits(fps)


        # Paw motion
        dlc_paw_front = (
            markers_body
            .loc[:, ('paw_front-right2', ['x','y','likelihood'])]  # select the 3 subcols
            .droplevel(0, axis=1)                                   # leave cols: x, y, likelihood
        )
        dlc_paw_back = (
            markers_body
            .loc[:, ('paw_back-right2', ['x','y','likelihood'])]  # select the 3 subcols
            .droplevel(0, axis=1)                                   # leave cols: x, y, likelihood
        )
        motion_frontpaw = body_processing.dlc_pointmotion(dlc_paw_front)
        motion_backpaw = body_processing.dlc_pointmotion(dlc_paw_back)

        # Paw angles
        angle_paws_front = body_processing.dlc_angle(
            markers_body['paw_front-right1'], markers_body['paw_front-right2'], markers_body['paw_front-right3'])
        angle_paws_back = body_processing.dlc_angle(
            markers_body['paw_back-right1'],  markers_body['paw_back-right2'],  markers_body['paw_back-right3'])

        # Stride and gait information
        rightpaws_fbdiff_raw, _  = body_processing.dlc_pointdistance(
            markers_body['paw_front-right2'], markers_body['paw_back-right2'])
        stride_freq = motion_processing.freq_analysis2(
            rightpaws_fbdiff_raw, fps)

        # Mouth motion
        dlc_mouth = (
            markers_body
            .loc[:, ('mouth', ['x','y','likelihood'])]  # select the 3 subcols
            .droplevel(0, axis=1)                                   # leave cols: x, y, likelihood
        )
        motion_mouth = body_processing.dlc_pointmotion(dlc_mouth)

        # Tail information
        angle_tail = body_processing.dlc_angle(
            markers_body['tail1'], markers_body['tail2'], markers_body['tail3'])

        # cylinder_mask = np.zeros([h, w])
        # cylinder_mask[int(np.nanpercentile(
        #     markers_body['paw_back-right1', 'y'].values, 99) + 30):, :int(w/3)] = 1
        # cylinder_motion = body_processing.cylinder_motion(
        #     video_file, cylinder_mask)

        idx = markers_body.index
        

        # Due to smoothing we might get different lebgths compared to the raw data.
        # Therefore, we just fill up the shorter rows with NaNs to match the raw data lengths. 
        body_raw = pd.concat([
            self.as_series(motion_frontpaw.raw_distance, 'PointMotion_FrontPaw', idx),
            self.as_series(motion_frontpaw.angles,       'AngleMotion_FrontPaw', idx),
            self.as_series(motion_mouth.raw_distance,    'PointMotion_Mouth', idx),
            self.as_series(motion_mouth.angles,          'AngleMotion_Mouth', idx),
            self.as_series(motion_backpaw.raw_distance,  'PointMotion_BackPaw', idx),
            self.as_series(motion_backpaw.angles,        'AngleMotion_BackPaw', idx),
            self.as_series(stride_freq, 'Frequency_BackPaw', idx),
            self.as_series(angle_tail.angle3,            'Angle_Tail_3', idx),
            self.as_series(angle_tail.slope,             'Angle_Tail', idx),
            self.as_series(angle_paws_front.angle3,      'Angle_Paws_Front_3', idx),
            self.as_series(angle_paws_front.slope,       'Angle_Paws_Front', idx),
            self.as_series(angle_paws_back.angle3,       'Angle_Paws_Back_3', idx),
            self.as_series(angle_paws_back.slope,        'Angle_Paws_Back', idx),
            self.as_series(-zscore(markers_body['tail1','y']), 'Tailroot_Level', idx),
        ], axis=1)


        # further process raw data and save
        cap.release()
        body_raw.to_hdf(out_file, key='body')
        print(f"saved mouseflow body analysis to {out_file}")


def runMF(dlc_dir=os.getcwd(),
          overwrite=False,
          dgp=True,
          conf_thresh=None,
          interpolation_limits_sec={
              'pupil': 2,
              'eyelid': 1,
          },
          smoothing_windows_sec={
              'PupilDiam': 1,
              'PupilMotion': 0.25,
              'eyelid': 0.1,
              'MotionEnergy': 0.25,
          },
          na_limit=0.25,
          faceregions_sizes=None,
          base_resolution=None,
          manual_anchor=None         
    ):

        cfg = MFConfig(dgp, conf_thresh,
                    interpolation_limits_sec, smoothing_windows_sec,
                    na_limit, faceregions_sizes,
                    base_resolution, manual_anchor, "RAFT", overwrite
        )

        mf = MouseFlow(dlc_dir, cfg)
        mf.run()

