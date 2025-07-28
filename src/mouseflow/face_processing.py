#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyTreadMouse functions to analyse camera data

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""
from typing import NamedTuple

import glob
import math
import os.path

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from mouseflow.utils.generic import smooth

plt.interactive(False)

# PUPIL
def pupilextraction(pupil_markers_xy_confident):
    pupil_circle = np.zeros(
        shape=(len(pupil_markers_xy_confident), 2), dtype=object)
    print("Fitting circle to pupil...")

    # 2D points from DLC MouseFace
    for i in tqdm(range(len(pupil_markers_xy_confident))):
        pupilpoints = np.float32(pupil_markers_xy_confident[i].reshape(6, 2))
        pupil_circle[i, :] = cv2.minEnclosingCircle(pupilpoints)

    # extract pupil centroid
    pupil_centre = np.asarray(tuple(pupil_circle[:, 0]), dtype=np.float32)
    pupil_x_raw = pd.Series(pupil_centre[:, 0])
    pupil_y_raw = pd.Series(pupil_centre[:, 1])

    # extract pupil movement
    pupil_xy = pd.DataFrame(
        np.array((pupil_x_raw, pupil_y_raw)).T, columns=['x', 'y'])
    pupil_xy[pupil_xy == 0] = np.nan
    pupil_xydist_raw = pd.Series(np.linalg.norm(pupil_xy.diff(), axis=1))

    # extract pupil diameter
    pupil_diam_raw = pd.Series(pupil_circle[:, 1], dtype=np.float32)

    return pd.DataFrame({'PupilX': pupil_x_raw, 
                         'PupilY': pupil_y_raw, 
                         'PupilMotion': pupil_xydist_raw, 
                         'PupilDiam': pupil_diam_raw})


# FACE REGIONS
def define_faceregions(dlc_face, facevid, dlc_file, manual_anchor=None, faceregions_sizes=None, base_resolution=(782, 582)):
    # checking on video
    facemp4 = cv2.VideoCapture(facevid)
    if not facemp4.isOpened():
        raise IOError(f"Cannot open video file: {facevid}")
    ret, frame = facemp4.read()
    facemp4.release()
    if not ret or frame is None:
        raise IOError(f"Cannot read first frame from: {facevid}")
    firstframe = frame[:, :, 0].astype(np.uint8)
    plt.imshow(firstframe, cmap='gray');

    # create empty canvas
    canvas = np.zeros(firstframe.shape, dtype=np.uint8)

    # define face anchors
    anchor_names = ['nosetip', 'forehead', 'mouthtip',
                    'chin', 'tearduct', 'eyelid2']
    face_anchor = pd.DataFrame(index=['x', 'y'], columns=anchor_names)
    for anchor_name in anchor_names:
        if manual_anchor and anchor_name in manual_anchor:
            x, y = manual_anchor[anchor_name]
            face_anchor.at['x', anchor_name] = x
            face_anchor.at['y', anchor_name] = y
            plt.scatter(x, y, c='yellow', s=40)
        else:
            if dlc_face[anchor_name, 'x'].isnull().mean() == 1:  # if only missing values, leave empty
                face_anchor.at['x', anchor_name] = np.nan
                face_anchor.at['y', anchor_name] = np.nan
            else:
                x = dlc_face[anchor_name, 'x'].mean()
                y = dlc_face[anchor_name, 'y'].mean()
                face_anchor.at['x', anchor_name] = x
                face_anchor.at['y', anchor_name] = y
                plt.scatter(dlc_face[anchor_name]['x'],
                            dlc_face[anchor_name]['y'], alpha=.002)
                plt.scatter(face_anchor[anchor_name]['x'],
                            face_anchor[anchor_name]['y'])
    face_anchor['eyelid_bottom'] = face_anchor['eyelid2'] # in DLC the keypoints are labeled with 1,2,3,etc., 
    face_anchor = face_anchor.astype(float)               # ensure numeric

    # compute scaling if sizes provided
    H, W = firstframe.shape
    w0, h0 = base_resolution
    scale_x = W / w0
    scale_y = H / h0

    def get_axes(region):
        """
        If provide manual faceregions_sizes, this returns the
        axes (width, height) for the given region in pixels, scaled
        from the base_resolution to the current video resolution.
        Otherwise returns None to trigger the automatic size.
        """
        if faceregions_sizes and region in faceregions_sizes:
            sx0, sy0 = faceregions_sizes[region]
            # scale from base_resolution to actual frame size
            return (sx0 * scale_x, sy0 * scale_y)
        return None

    def create_mask(centre, axes, angle):
        sx, sy = map(int, axes)
        cv2.ellipse(firstframe, centre, (sx, sy), angle, 0, 360, (255,0,0), 4)
        mask = cv2.ellipse(canvas.copy(), centre, (sx, sy), angle, 0, 360, (1,), -1)
        return mask.astype(bool)
    
    def get_pt(name):
        arr = face_anchor.loc[['x','y'], name].values
        return None if np.isnan(arr).all() else arr

    nose_pt    = get_pt('nosetip')
    mouth_pt   = get_pt('mouthtip')
    tear_pt    = get_pt('tearduct')
    forehead_pt= get_pt('forehead')
    chin_pt    = get_pt('chin')
    eyelid_pt  = get_pt('eyelid_bottom')

    # whisker inference
    if nose_pt is None or mouth_pt is None or tear_pt is None:
        mask_whiskers = canvas.astype(bool)
    else:
        centre_whiskers = tuple(np.round(np.vstack([
            nose_pt,
            mouth_pt,
            tear_pt]).mean(axis=0)).astype(int))
        a = np.linalg.norm(mouth_pt - tear_pt)
        b = np.linalg.norm(tear_pt  - nose_pt)
        c = np.linalg.norm(nose_pt  - mouth_pt)
        s = 0.5*(a+b+c)
        whisker_r = math.sqrt(s*(s-a)*(s-b)*(s-c)) / s
        axes = get_axes('whiskers') or (whisker_r, whisker_r)
        mask_whiskers = create_mask(centre_whiskers, axes, angle=0)
    # nose inference
    if nose_pt is None:
        mask_nose = canvas.copy()
    else:
        centre_nose = tuple((nose_pt + np.array([0.05*nose_pt[0], -0.1*nose_pt[1]])).round().astype(int))
        axes = get_axes('nose') or (whisker_r*2/3, whisker_r*1/2)
        mask_nose = create_mask(centre_nose, axes, angle=-60.0)
    # mouth inference ellipse
    if mouth_pt is None or chin_pt is None:
        mask_mouth = canvas.astype(bool)
    else:
        centre_mouth = tuple(np.round(mouth_pt + (chin_pt - mouth_pt)/3).astype(int))
        angle_mouth = math.degrees(math.atan2(chin_pt[1]  - mouth_pt[1], chin_pt[0]  - mouth_pt[0]))
        dist = np.hypot(chin_pt[0]  - mouth_pt[0], chin_pt[1]  - mouth_pt[1])
        axes = get_axes('mouth') or (dist, dist/4)
        mask_mouth = create_mask(centre_mouth, axes, angle_mouth)
    # cheek inference ellipse
    if chin_pt is None or eyelid_pt is None:
        mask_cheek = canvas.astype(bool)
    else:
        centre_cheek = tuple(np.round(eyelid_pt + (chin_pt - eyelid_pt)/2).astype(int))
        dist = np.linalg.norm(chin_pt - eyelid_pt) 
        axes = get_axes('cheek') or ((2/3)*dist, (1/3)*dist)
        mask_cheek = create_mask(centre_cheek, axes, angle=0)
    masks = [mask_nose, mask_whiskers, mask_mouth, mask_cheek]
    if dlc_file:
        plt.imshow(firstframe, cmap='gray')
        plt.savefig(dlc_file[:-3] + "_face_regions.png");
    plt.close('all');

    return masks, face_anchor


# Calculating optical flow and motion energy
def facemotion(videopath, masks, videoslice=[]):
    print(f"Calculating optical flow and motion energy for video {videopath}...")
    facemp4 = cv2.VideoCapture(videopath)
    if videoslice:
        print("Processing slice from {} to {}...".format(
            videoslice[0], videoslice[-1]))
        facemp4.set(1, videoslice[0])
        framelength = videoslice[1]-videoslice[0]
    else:
        framelength = int(facemp4.get(7))
    _, current_frame = facemp4.read()
    previous_frame = current_frame
    gpu_masks = []
    maskpx = []
    masks = [m.astype('float32') for m in masks]
    for m in range(len(masks)):
        gpu_mask = cv2.cuda_GpuMat()
        gpu_mask.upload(masks[m])
        gpu_masks.append(gpu_mask)
        masks[m][masks[m] == 0] = np.nan
        maskpx.append(np.nansum(masks[m]))

    frame_diff = np.empty((framelength, 4))
    frame_diffmag = np.empty((framelength, 4))
    frame_diffang = np.empty((framelength, 4))
    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(numLevels=5, pyrScale=.5, fastPyramids=True, winSize=25,
                                                    numIters=3, polyN=5, polySigma=1.2, flags=0)

    i = 0
    with tqdm(total=framelength) as pbar:
        while facemp4.isOpened():
            current_frame_gray = cv2.cvtColor(
                current_frame, cv2.COLOR_BGR2GRAY)
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(current_frame_gray)

            previous_frame_gray = cv2.cvtColor(
                previous_frame, cv2.COLOR_BGR2GRAY)
            gpu_previous = cv2.cuda_GpuMat()
            gpu_previous.upload(previous_frame_gray)
            flow = gpu_flow.calc(gpu_frame, gpu_previous, None)
            gpu_flow_x = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
            gpu_flow_y = cv2.cuda_GpuMat(flow.size(), cv2.CV_32FC1)
            cv2.cuda.split(flow, [gpu_flow_x, gpu_flow_y])
            gpu_mag, gpu_ang = cv2.cuda.cartToPolar(
                gpu_flow_x, gpu_flow_y, angleInDegrees=True)
            gpu_frame32 = gpu_frame.convertTo(cv2.CV_32FC1, gpu_frame)
            gpu_previous32 = gpu_previous.convertTo(
                cv2.CV_32FC1, gpu_previous)
            for index, (mask, gpu_mask, px) in enumerate(zip(masks, gpu_masks, maskpx)):
                mag_mask = cv2.cuda.multiply(gpu_mag, gpu_mask)
                ang_mask = cv2.cuda.multiply(gpu_ang, gpu_mask)
                frame_mask = cv2.cuda.multiply(gpu_frame32, gpu_mask)
                previous_mask = cv2.cuda.multiply(gpu_previous32, gpu_mask)
                frame_diffmag[i, index] = cv2.cuda.absSum(mag_mask)[0] / px
                frame_diffang[i, index] = cv2.cuda.absSum(ang_mask)[0] / px
                frame_diff[i, index] = cv2.cuda.absSum(
                    cv2.cuda.absdiff(frame_mask, previous_mask))[0] / px
            pbar.update(1)
            i += 1
            previous_frame = current_frame.copy()
            _, current_frame = facemp4.read()
            if current_frame is None or (videoslice and len(frame_diff) > len(videoslice)-1):
                break
    facemp4.release()

    motion = pd.DataFrame(np.hstack([frame_diff, frame_diffmag, frame_diffang]),
                            columns=['MotionEnergy_Nose', 'MotionEnergy_Whiskerpad', 'MotionEnergy_Mouth', 'MotionEnergy_Cheek',
                                    'OFmag_Nose', 'OFmag_Whiskerpad', 'OFmag_Mouth', 'OFmag_Cheek',
                                    'OFang_Nose', 'OFang_Whiskerpad', 'OFang_Mouth', 'OFang_Cheek'])
    return motion


# Calculating motion energy
def facemotion_nocuda(videopath, masks, videoslice=[]):
    print(f"Calculating motion energy for video {videopath}...")
    facemp4 = cv2.VideoCapture(videopath)
    if videoslice:
        print("Processing slice from {} to {}...".format(
            videoslice[0], videoslice[-1]))
        facemp4.set(1, videoslice[0])
        framelength = videoslice[1]-videoslice[0]
    else:
        framelength = int(facemp4.get(7))
    _, current_frame = facemp4.read()
    previous_frame = current_frame
    masks = [m.astype('float32') for m in masks]
    for m in range(len(masks)):
        masks[m][masks[m] == 0] = np.nan

    frame_diff = np.empty((framelength, 4))

    i = 0
    with tqdm(total=framelength) as pbar:
        while facemp4.isOpened():
            current_frame_gray = cv2.cvtColor(
                current_frame, cv2.COLOR_BGR2GRAY)
            previous_frame_gray = cv2.cvtColor(
                previous_frame, cv2.COLOR_BGR2GRAY)

            for index, mask in enumerate(masks):
                frame_diff[i, index] = np.nanmean(cv2.absdiff(
                    current_frame_gray * mask, previous_frame_gray * mask))
            pbar.update(1)
            i += 1
            previous_frame = current_frame.copy()
            ret, current_frame = facemp4.read()
            if current_frame is None or (videoslice and len(frame_diff) > len(videoslice)-1):
                break
    facemp4.release()

    motion = pd.DataFrame(frame_diff,
                          columns=['MotionEnergy_Nose', 'MotionEnergy_Whiskerpad',
                                   'MotionEnergy_Mouth', 'MotionEnergy_Cheek',])

    return motion
