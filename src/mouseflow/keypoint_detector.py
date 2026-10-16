"""
Author: @Jakob Faust
Date: 10.10.2026

Wrapper class for body and face keypoint models.
Starts detection process of keypoints.
"""


from pathlib import Path
from mouseflow.utils.preprocess_video import flip_vid, crop_vid
from mouseflow.apply_models import LPDetector, DLCDetector, download_models


class KeypointDetector():
    def __init__(self, keypoint_engine_face='DLC', filetype_face='avi',
                 crop_box_face=None, flip_input_face=False, 
                 flip_input_body=False, keypoint_model_weights_face=None,
                 keypoint_model_weights_body=None, models_dir=None, 
                 keypoint_engine_body='LP', filetype_body='mp4',
                 crop_box_body=None, overwrite=False, save_labled_videos=True):

        self.keypoint_engine_face = keypoint_engine_face
        self.crop_box_face = crop_box_face
        self.flip_input_face = flip_input_face
        self.keypoint_model_weights_face = keypoint_model_weights_face
        self.models_dir = models_dir
        self.keypoint_engine_body = keypoint_engine_body
        self.filetype_face = filetype_face
        self.filetype_body = filetype_body
        self.flip_input_body = flip_input_body
        self.crop_box_body = crop_box_body
        self.keypoint_model_weights_body = keypoint_model_weights_body
        self.overwrite = overwrite
        self.save_labled_videos = save_labled_videos
        if self.models_dir is None:
            self.models_dir = Path.cwd() / "mf_models"
                    
        self.model_paths = download_models(self.models_dir)


    def detect_facial_landmarks(self, vid_dir, cfg=None, include_key=None, n_videos=-1):
        if cfg is None or self.keypoint_model_weights_face is None:
            if self.keypoint_engine_face == 'DLC':
                cfg = str(self.model_paths['cfg_face_dlc'])
                self.keypoint_model_weights_face = str(self.model_paths['model_face_dlc'])
            else:
                raise ValueError("Only DLC engine supported for face landmark detection")

        vid_path = Path(vid_dir)
        
        if include_key is None:
            facefiles = [str(p) for p in vid_path.glob(f"*{self.filetype_face}")]
        else:
            facefiles = [str(p) for p in vid_path.glob(f"*{include_key}*{self.filetype_face}")]


        facefiles = [f for f in facefiles if 'labeled' not in f]
        facefiles = [f for f in facefiles if '_cropped' not in f]  # sort out already cropped videos
        if self.crop_box_face:
            facefiles_cropped = []
            for vid in facefiles:
                facefiles_cropped.append(crop_vid(vid, self.crop_box_face))
            facefiles = facefiles_cropped

        facefiles = [f for f in facefiles if '_flipped' not in f]  # sort out already flipped videos
        if self.flip_input_face:
            facefiles_flipped = []
            for vid in facefiles:
                facefiles_flipped.append(flip_vid(vid, horizontal=True))
            facefiles = facefiles_flipped

        if not facefiles:
            print(f'No face {self.filetype_face} files found to process in {vid_dir}')
            return None
        try:
            if n_videos > 0 and n_videos < len(facefiles):
                facefiles = facefiles[:n_videos]
                print(f'only processing first {n_videos} in {vid_dir}')
        except ValueError:
            print("only positive numbers allowed for 'n_videos'")


        if vid_path.is_dir():
            dir_out = vid_path / 'mouseflow'
        else:
            dir_out = vid_path.parent / 'mouseflow'

        dir_out.mkdir(parents=True, exist_ok=True)

        if cfg and self.keypoint_model_weights_face:
            for facefile in facefiles:
                face_path = Path(facefile)
                base = face_path.stem
                
                out_exists = list(dir_out.glob(f"{base}*.h5")) + list(dir_out.glob(f"{base}*.csv"))

                if out_exists and not self.overwrite:
                    print(f"Skipping {face_path.name} (already labelled: {out_exists[0]}).")
                    continue

                print("Applying FACE model:", cfg, "weights:", self.keypoint_model_weights_face)
                det = DLCDetector(cfg, self.keypoint_model_weights_face, shuffle=2)
                    
                det.detect_keypoints(str(face_path), str(dir_out), self.save_labled_videos, self.overwrite)

    def detect_body_keypoints(self, vid_dir, cfg=None, include_key=None, n_videos=-1):
        if cfg is None or self.keypoint_model_weights_body is None:
            if self.keypoint_engine_body == 'DLC':
                cfg = str(self.model_paths['cfg_body_dlc'])
                self.keypoint_model_weights_body = str(self.model_paths['model_body_dlc'])
            elif self.keypoint_engine_body == 'LP':
                cfg = str(self.model_paths['cfg_body_lp'])
                self.keypoint_model_weights_body = str(self.model_paths['model_body_lp']) 
            else:
                raise ValueError("please use either LP or DLC for 'body_model'")

        vid_path = Path(vid_dir)
        
        if include_key is None:
            body_files = [str(p) for p in vid_path.glob(f"*{self.filetype_body}")]
        else:
            body_files = [str(p) for p in vid_path.glob(f"*{include_key}*{self.filetype_body}")]

        body_files = [f for f in body_files if 'labeled' not in f]
        body_files = [f for f in body_files if '_cropped' not in f]  # sort out already cropped videos
        if self.crop_box_body:
            body_files_cropped = []
            for vid in body_files:
                body_files_cropped.append(crop_vid(vid, self.crop_box_body))
            body_files = body_files_cropped

        body_files = [f for f in body_files if '_flipped' not in f]  # sort out already flipped videos
        if self.flip_input_body:
            body_files_flipped = []
            for vid in body_files:
                body_files_flipped.append(flip_vid(vid, horizontal=True))
            body_files = body_files_flipped

        if not body_files:
            print(f'No face {self.filetype_body} files found to process in {vid_dir}')
            return None
        try:
            if n_videos > 0 and n_videos < len(body_files):
                body_files = body_files[:n_videos]
                print(f'only processing first {n_videos} in {vid_dir}')
        except ValueError:
            print("only positive numbers allowed for 'n_videos'")


        if vid_path.is_dir():
            dir_out = vid_path / 'mouseflow'
        else:
            dir_out = vid_path.parent / 'mouseflow'

        dir_out.mkdir(parents=True, exist_ok=True)

        if cfg and self.keypoint_model_weights_face:
            for bodyfile in body_files:
                body_file_path = Path(bodyfile)
                base = body_file_path.stem
                
                out_exists = list(dir_out.glob(f"{base}*.h5")) + list(dir_out.glob(f"{base}*.csv"))

                if out_exists and not self.overwrite:
                    print(f"Skipping {body_file_path.name} (already labelled: {out_exists[0]}).")
                    continue

                print("Applying BODY model:", cfg, "weights:", self.keypoint_model_weights_face)
                if self.keypoint_engine_body == 'LP':
                    det = LPDetector(cfg, self.keypoint_model_weights_body)
                else:
                    det = DLCDetector(cfg, self.keypoint_model_weights_body, shuffle=3)
                det.detect_keypoints(str(bodyfile), str(dir_out), self.save_labled_videos, self.overwrite)
       
