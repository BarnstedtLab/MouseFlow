#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from pathlib import Path

from mouseflow.face_processing import FaceAnalyzer
from mouseflow.body_processing import BodyAnalyzer
from mouseflow.keypoint_detector import KeypointDetector


def get_files2analyse(keypoint_dir, overwrite):
    '''
    Returns face and body files hich can be found in dlc_dir and should be analysed.
    The face files must match the pattern '*DLC*MouseFace*.h5'.
    The body files must match the pattern '*DLC*MouseBody*.h5'.
    '''
    if keypoint_dir is None:
        raise RuntimeError("No DLC directory defined.")
    d = Path(keypoint_dir)

    face_raw = sorted(d.glob('*DLC*MouseFace*.h5'))
    body_raw = sorted(d.glob('*DLC*MouseBody*.h5'))

    def needs_processing(p: Path) -> bool:
        if p.name.lower().endswith("_mouseflow.h5"):
            return False
        mf = p.with_name(p.stem + '_mouseflow.h5')
        if mf.exists() and not overwrite:
            print(f"skipping file {p.name} as it is already analysed!") 
        return overwrite or not mf.exists()

    face_files = [p for p in face_raw if needs_processing(p)]
    body_files = [p for p in body_raw if needs_processing(p)]

    return {'face_files': face_files, 'body_files': body_files}

def analyse_face(keypoint_dir, overwrite=False, opticalflow_engine='DIS', save_vectorfields=False):
    face_files = get_files2analyse(keypoint_dir=keypoint_dir, overwrite=overwrite)['face_files']
    if not face_files:
        print(f"No face files found to process in {keypoint_dir}.")
    face_analyzer = FaceAnalyzer(cfg=None, opticalflow_enigne=opticalflow_engine, save_vectorfield=save_vectorfields)
    for current_file in face_files:
        face_analyzer.analyse(current_file, landmark_dir=keypoint_dir)

def analyse_body(keypoint_dir, overwrite=False):
    body_files = get_files2analyse(keypoint_dir=keypoint_dir, overwrite=overwrite)['body_files']
    if not body_files:
        print(f"No body files found to process in {keypoint_dir}.")
    body_analyzer = BodyAnalyzer()
    for current_file in body_files:
        body_analyzer.analyse(current_file, keypoint_dir=keypoint_dir)

def pose_detection(video_dir, models_dir=None, keypoint_engine_face='DLC', filetype_face='mp4', crop_box_face=None, face_facing='left', 
                     body_facing='right', keypoint_model_weights_face=None, keypoint_model_weights_body=None,
                     keypoint_engine_body='LP', filetype_body='mp4', crop_box_body=None,
                     overwrite=False, save_labled_videos=True, face_videokey=None, body_videokey=None):

    kp_detector = KeypointDetector(
        keypoint_engine_face=keypoint_engine_face, filetype_face=filetype_face, crop_box_face=crop_box_face,
        flip_input_face=(face_facing!='left'), flip_input_body=(body_facing!='right'), 
        keypoint_model_weights_face=keypoint_model_weights_face,
        keypoint_model_weights_body=keypoint_model_weights_body, models_dir=models_dir,
        keypoint_engine_body=keypoint_engine_body, filetype_body=filetype_body, crop_box_body=crop_box_body,
        overwrite=overwrite, save_labled_videos=save_labled_videos
        )

    print('keypoint detector initialised!')

    if face_videokey is None and body_videokey is None:
        print("not sure weather to analyse body or face videos or both. Please provide either 'face_videokey'" \
        " or 'body_videokey'.")
        return None
    elif type(face_videokey) == bool or type(body_videokey) == bool:
        if face_videokey and not body_videokey:
            kp_detector.detect_facial_landmarks(video_dir, cfg=None, include_key=None)
        elif not face_videokey and body_videokey:
            kp_detector.detect_body_keypoints(video_dir, cfg=None, include_key=None)
    elif type(face_videokey) == str and type(body_videokey) == str:
        kp_detector.detect_facial_landmarks(video_dir, cfg=None, include_key=face_videokey)
        kp_detector.detect_body_keypoints(video_dir, cfg=None, include_key=body_videokey)
    else:
        print("Please provide either 'face_videokey = True' if all videos you want to analyse are face videos or " \
                "'body_videokey = True' if all videos you want to analyse are body videos." \
                "If both videos are stored in the same directory, provide unique identifiers like 'body_videokey = 'body''" \
                "and face_videokey = 'face' which are present in the file names.")
        return None
    
