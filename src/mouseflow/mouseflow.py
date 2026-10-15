# -*- coding: utf-8 -*-
"""
Author: @Jakob Faust
Date: 15.10.2026

This file holds all main functions that are exposed to the user:
 - pose_body() : detects bodily keypoints
 - pose_body() : detects facial keypoints (used for mask generation in optical flow)
 - analyse_face() : analyses facial kinematics videos using optical flow
 - analyse_body() : analyses body/gait kinematics based on keypoint tracking results
"""

from pathlib import Path

from mouseflow.face_processing import FaceAnalyzer
from mouseflow.body_processing import BodyAnalyzer
from mouseflow.keypoint_detector import KeypointDetector
from mouseflow.utils.filehandler import get_files2analyse


def analyse_face(keypoint_dir, overwrite=False, opticalflow_engine='DIS', save_vectorfields=False):
    face_files = get_files2analyse(keypoint_dir=keypoint_dir, overwrite=overwrite, analysis_type='face')
    if not face_files:
        print(f"No face files found to process in {keypoint_dir}.")
    face_analyzer = FaceAnalyzer(cfg=None, opticalflow_enigne=opticalflow_engine, save_vectorfield=save_vectorfields)
    for current_file in face_files:
        face_analyzer.analyse(current_file, landmark_dir=keypoint_dir)

def analyse_body(keypoint_dir, overwrite=False):
    body_files = get_files2analyse(keypoint_dir=keypoint_dir, overwrite=overwrite, analysis_type='body')
    if not body_files:
        print(f"No body files found to process in {keypoint_dir}.")
    body_analyzer = BodyAnalyzer()
    for current_file in body_files:
        body_analyzer.analyse(current_file, keypoint_dir=keypoint_dir)

def pose_face(video_dir, models_dir=None, keypoint_engine_face='DLC', filetype_face='mp4', 
              crop_box_face=None, face_facing='left', keypoint_model_weights_face=None, 
              overwrite=False, save_labled_videos=True, face_videokey=None):
    """Detects facial landmarks in the specified videos."""
    
    kp_detector = KeypointDetector(
        keypoint_engine_face=keypoint_engine_face, filetype_face=filetype_face, 
        crop_box_face=crop_box_face, flip_input_face=(face_facing!='left'), 
        keypoint_model_weights_face=keypoint_model_weights_face, models_dir=models_dir,
        overwrite=overwrite, save_labled_videos=save_labled_videos,
        # Default body arguments to satisfy KeypointDetector's __init__
        keypoint_engine_body='LP', filetype_body='mp4', crop_box_body=None, 
        flip_input_body=False, keypoint_model_weights_body=None
    )

    print('Keypoint detector initialised for face!')
    
    # If face_videokey is a string, use it to filter files. If it's a boolean, process all.
    include_key = face_videokey if isinstance(face_videokey, str) else None
    kp_detector.detect_facial_landmarks(video_dir, cfg=None, include_key=include_key)


def pose_body(video_dir, models_dir=None, keypoint_engine_body='LP', filetype_body='mp4', 
              crop_box_body=None, body_facing='right', keypoint_model_weights_body=None, 
              overwrite=False, save_labled_videos=True, body_videokey=None):
    """Detects body keypoints in the specified videos."""
    
    kp_detector = KeypointDetector(
        keypoint_engine_body=keypoint_engine_body, filetype_body=filetype_body, 
        crop_box_body=crop_box_body, flip_input_body=(body_facing!='right'), 
        keypoint_model_weights_body=keypoint_model_weights_body, models_dir=models_dir,
        overwrite=overwrite, save_labled_videos=save_labled_videos,
        # Default face arguments to satisfy KeypointDetector's __init__
        keypoint_engine_face='DLC', filetype_face='mp4', crop_box_face=None, 
        flip_input_face=False, keypoint_model_weights_face=None
    )

    print('Keypoint detector initialised for body!')
    
    # If body_videokey is a string, use it to filter files. If it's a boolean, process all.
    include_key = body_videokey if isinstance(body_videokey, str) else None
    kp_detector.detect_body_keypoints(video_dir, cfg=None, include_key=include_key)
    
