#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pyTreadMouse functions to analyse camera data

Created on Wed May  8 14:31:51 2019
@author: Oliver Barnstedt
"""


import glob
import math
import os.path

import cv2
import flow_vis  # visualisation from https://github.com/tomrunia/OpticalFlow_Visualization
import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats.mstats import zscore
from tqdm import tqdm

plt.interactive(False)

# path_vid_face = '/media/oliver/Oliver_SSD1/Musall/2202_cam1_20211213.avi'
# path_dlc_face = '/media/oliver/Oliver_SSD1/Musall/mouseflow/2202_cam1_20211213DeepCut_resnet50_MouseFaceAug21shuffle1_1030000.h5'
# path_mf = '/media/oliver/Oliver_SSD1/Musall/mouseflow/2202_cam1_20211213DeepCut_resnet50_MouseFaceAug21shuffle1_1030000_analysis.h5'
path_vid_face = '/media/oliver/Oliver_SSD1/Ziyan/Basler_acA1920-150um__40032679__20230413_113011645_crop.mp4'
path_dlc_face = '/media/oliver/Oliver_SSD1/Ziyan/mouseflow/Basler_acA1920-150um__40032679__20230413_113011645_cropDeepCut_resnet50_MouseFaceAug21shuffle1_1030000.h5'
path_mf = '/media/oliver/Oliver_SSD1/Ziyan/mouseflow/Basler_acA1920-150um__40032679__20230413_113011645_cropDeepCut_resnet50_MouseFaceAug21shuffle1_1030000_analysis.h5'
startframe=500
cols_to_plot=['PupilDiam', 'PupilX', 'MotionEnergy_Mouth', 'OFang_Whiskerpad', 'OFang_Nose']
generate_frames=1000
blend_gray_optflow=0.5
smooth_data=15
dlc_conf_thresh=0.99
use_dark_background = True
resultsdir=False

# preallocate empty array and assign slice by chrisaycock
def shift5(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result


def create_labeled_video_face(path_vid_face, path_dlc_face, path_mf,
                              generate_frames=1000, startframe=500, use_dark_background=True, resultsdir=False,
                              cols_to_plot=['PupilDiam', 'PupilX', 'MotionEnergy_Mouth', 'OFang_Whiskerpad', 'OFang_Nose'],
                              blend_gray_optflow=0.5, smooth_data=15, dlc_conf_thresh=None, processing_stage=None, make_video=True):                        
    facemp4 = cv2.VideoCapture(path_vid_face)
    facemp4.set(cv2.CAP_PROP_POS_FRAMES, startframe) # jump straight to the first frame wanted (explicit constant = clearer than “1”)
    FaceCam_FPS = facemp4.get(cv2.CAP_PROP_FPS)
    index = startframe
    if isinstance(generate_frames, int): # it's is True 
        # give an **exact count** → stop after N frames
        lastframe = startframe + generate_frames # track the last frame not just the count
        vidlength = generate_frames
    elif generate_frames==True:
        # special flag: label the *whole* video
        lastframe = int(facemp4.get(cv2.CAP_PROP_FRAME_COUNT)) # total frames
        index = 0
        vidlength = lastframe # same number as total frames
    elif len(generate_frames)==2:
        # give a [start, end) slice → reposition and compute lengths
        facemp4.set(1, generate_frames[0]) # jump to slice start
        lastframe = generate_frames[1] # absolute end frame
        index = generate_frames[0] # absolute start frame
        vidlength = lastframe - index # number of frames in the slice
    print("Labelling face video...")
    ret, current_frame = facemp4.read()
    previous_frame = current_frame

    gpu_flow = cv2.cuda_FarnebackOpticalFlow.create(numLevels=5, pyrScale=.5, fastPyramids=True, winSize=25,
                                                        numIters=3, polyN=5, polySigma=1.2, flags=0)

    # Load DLC points and face data
    # get rid of pupil cos it is not a seperate dataframe with a key
    dlc_face = pd.read_hdf(path_dlc_face)
    with h5py.File(path_mf, "r") as hfg: # auto-closes on exit
        facemasks = hfg["facemasks"][:].astype("float32")
    face = pd.read_hdf(path_mf, 'face')
    face_anchor = pd.read_hdf(path_mf, 'face_anchor')
    

    # Prepare layout
    plt.close("all")
    # plt.ioff()  # hide figures
    plt.ion()  # show figures
    fig, axd = plt.subplot_mosaic([['upper left', 'upper centre', 'upper right'],
                                ['bottom', 'bottom', 'bottom']], figsize=(14, 8), dpi=80)
    if use_dark_background:
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    if not resultsdir:
        resultsdir = os.path.join(os.path.dirname(path_dlc_face), 'mouseflow_'+os.path.basename(path_vid_face).split('.')[0])
    if not os.path.exists(resultsdir):
        os.makedirs(resultsdir, exist_ok=True) # suppresses the FileExistsError if exists

    showseconds = 10
    showtimepoints = int(FaceCam_FPS) * showseconds
    midpoint = int(showtimepoints/2)
    idx = pd.IndexSlice
    # Select which processing stage we want to visualise 
    if processing_stage in ('raw', 'interpolated'):
        # pull raw/interpolated columns, then smooth + z-score
        data = face.loc[:, idx[processing_stage, cols_to_plot]]
        face_data = data.apply(zscore).rolling(smooth_data, center=True, min_periods=1).mean()
    elif processing_stage == 'smooth':
        # already smoothed → just z-score
        data = face.loc[:, idx['smooth', cols_to_plot]]
        face_data = data.apply(zscore)
    elif processing_stage == 'zscore':
        # already z-scored → just smooth for display
        data = face.loc[:, idx['zscore', cols_to_plot]]
        cols_to_smooth = ['OFmag_Mouth', 'OFmag_Cheek']
        cols_to_smooth_full = [('zscore', c) for c in cols_to_smooth]
        cols_unsmoothed_full = [col for col in data.columns if col not in cols_to_smooth_full]
        smoothed = data[cols_to_smooth_full].rolling(smooth_data, center=True, min_periods=1).mean()
        unsmoothed = data[cols_unsmoothed_full]
        face_data     = pd.concat([smoothed, unsmoothed], axis=1)[data.columns]
    else:
        raise ValueError(f"Unknown stage {processing_stage}")
    trace_spacing = 3
    mins = face_data.quantile(0.01) - np.arange(face_data.shape[1]) * trace_spacing
    maxs = face_data.quantile(0.99) - np.arange(face_data.shape[1]) * trace_spacing
    minvalue, maxvalue = mins.min(), maxs.max()
    cmap = plt.get_cmap('Dark2')

    # Extra helper vars
    # the order of regions matches the order of facemarks so that the bottom traces match the upper left plot
    region_order = ['Nose', 'Whiskerpad', 'Mouth', 'Cheek'] # same order as masks
    region_to_idx = {}
    for idx, col in enumerate(cols_to_plot):
        for region in region_order:
            if col.lower().endswith(region.lower()): # cos the 'cols_to_plot' is named: OFmag_Nose....
                region_to_idx[region] = idx # remember colour slot for that region
                break # stop scanning that col once matched
    fallback_idx = 0
    # identify pupil landmark indices inside DLC MultiIndex
    pupil_pts = []
    cols = dlc_face.columns
    num_pts = len(cols) // 3                
    for i in range(num_pts):
        xcol = cols[i*3]                    # first col of each triplet
        bodypart = xcol[1] if isinstance(xcol, tuple) else xcol # if the DLC MultiIndex column (example: ('DLC_resnet50', 'pupil1', 'x')) then take the bodypart name (2nd element)
        if bodypart.lower().startswith('pupil'):
            pupil_pts.append(i)

    with tqdm(total=vidlength) as pbar:
        while facemp4.isOpened():
            # == BOTTOM PANEL ==
            # x-axis setup
            axd['bottom'].cla()
            axd['bottom'].set_xlim(0, showtimepoints)
            # generate ticks for every fps
            frame_ticks = np.arange(0, showtimepoints + 1, FaceCam_FPS)
            # convert frame indices to “seconds from center” so labels
            sec_ticks = (frame_ticks - midpoint) / FaceCam_FPS
            axd['bottom'].set_xticks(frame_ticks)
            axd['bottom'].set_xticklabels([f"{s:.0f}" for s in sec_ticks])      
            # y-axis setup
            pad = (maxvalue - minvalue) * 0.15           # 30 % head/foot-value
            axd['bottom'].set_ylim(minvalue - pad, maxvalue + pad)
            axd['bottom'].set_yticks(-np.arange(len(cols_to_plot)) * trace_spacing)
            axd['bottom'].set_yticklabels(cols_to_plot, fontsize=15, fontweight='bold')
            # data plotting
            start = max(index - midpoint, face_data.index.min()) # take the first valid frame
            end = min(index + midpoint, face_data.index.max())
            window = face_data.loc[start:end]
            # pad with NaNs if we’re near start/end of video so the plotted window is always `showtimepoints` long.
            if len(window) < showtimepoints:
                top_pad = max(0, midpoint - (index - start))
                bot_pad = showtimepoints - len(window) - top_pad
                pad_top = np.full((top_pad, window.shape[1]), np.nan)
                pad_bot = np.full((bot_pad, window.shape[1]), np.nan)
                window = pd.DataFrame(
                    np.vstack([pad_top, window.values, pad_bot]),
                    columns=window.columns)
            # plot each trace
            for i, col in enumerate(cols_to_plot):
                axd['bottom'].axhline(-i * trace_spacing, linestyle='--',
                                    color=cmap(i), alpha=0.5, linewidth=1.2)
                axd['bottom'].plot(window.values[:, i] - i * trace_spacing, color=cmap(i))
            axd['bottom'].axvline(midpoint, color='white' if use_dark_background else 'black')
            axd['bottom'].set_xlabel('Time [sec]')

            # == TOP PANEL ==
            # plot optical flow face frame
            current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            axd['upper left'].cla()
            previous_frame_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(current_frame_gray)
            gpu_previous = cv2.cuda_GpuMat()
            gpu_previous.upload(previous_frame_gray)
            flow = gpu_flow.calc(gpu_frame, gpu_previous, None)
            flow_color = flow_vis.flow_to_color(flow.download(), convert_to_bgr=True)
            if blend_gray_optflow:
                dst = cv2.addWeighted(current_frame, blend_gray_optflow, flow_color, 1-blend_gray_optflow, 0)
            else:
                dst = flow_color
            axd['upper left'].imshow(dst)
            axd['upper left'].axis('off')
            # plt.text(10, 30, str(params['paths']['Data_FaceCam'][-22:]), color='w', size=8)
            axd['upper left'].text(10, 30, "%06d" % index, color='w')

            # plot face regions
            for mask, region in zip(facemasks, region_order):
                colour_idx = region_to_idx.get(region, fallback_idx)
                axd['upper left'].contour(mask, colors=[cmap(colour_idx)], linestyles='--', linewidths=2)

            # Raw face with calculated anchor points
            axd['upper centre'].cla()
            axd['upper centre'].imshow(current_frame_gray, cmap='gray')
            axd['upper centre'].axis('off')
            # plt.text(10, 30, str(params['paths']['Data_FaceCam'][-22:]), color='w', size=8)
            axd['upper centre'].text(10, 30, "%06d" % index, color='w')

            # plot fix points
            for key in face_anchor:
                axd['upper centre'].scatter(*face_anchor[key], s=200, color='w', alpha=0.5)

            # plot skeleton
            axd['upper centre'].plot([face_anchor.nosetip[0], face_anchor.forehead[0]], [face_anchor.nosetip[1], face_anchor.forehead[1]], color='w')
            axd['upper centre'].plot([face_anchor.nosetip[0], face_anchor.mouthtip[0]], [face_anchor.nosetip[1], face_anchor.mouthtip[1]], color='w')
            axd['upper centre'].plot([face_anchor.nosetip[0], face_anchor.tearduct[0]], [face_anchor.nosetip[1], face_anchor.tearduct[1]], color='w')
            axd['upper centre'].plot([face_anchor.mouthtip[0], face_anchor.tearduct[0]], [face_anchor.mouthtip[1], face_anchor.tearduct[1]], color='w')
            axd['upper centre'].plot([face_anchor.eyelid_bottom[0], face_anchor.chin[0]], [face_anchor.eyelid_bottom[1], face_anchor.chin[1]], color='w')

            # plot dynamic DLC points whole face
            for i in range(num_pts - 6):
                if dlc_face.values[index, i*3+2] > dlc_conf_thresh:
                    axd['upper centre'].scatter(dlc_face.values[index, i*3], dlc_face.values[index, i*3+1], alpha=0.7, s=100, color='w')
            axd['upper centre'].scatter(face.loc[index, ('raw','PupilX')], face.loc[index, ('raw','PupilY')], alpha=0.7, s=30, color='w')
            axd['upper centre'].set_xlim(0, facemp4.get(cv2.CAP_PROP_FRAME_WIDTH))
            axd['upper centre'].set_ylim(facemp4.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)

            # Eye-detail crop
            axd['upper right'].cla()
            meanx = face[('smooth','PupilX')].mean()
            pupilmeanx = np.float64(meanx).round().astype(int)
            meany = face[('smooth','PupilY')].mean()
            pupilmeany = np.float64(meany).round().astype(int)
            eye_w, eye_h = 240, 180
            x0, y0 = pupilmeanx - eye_w//2, pupilmeany - eye_h//2
            eye_patch = current_frame_gray[y0:y0+eye_h, x0:x0+eye_w]
            axd['upper right'].imshow(eye_patch, cmap='gray')
            axd['upper right'].axis('off')

            # Overlay all in‐patch DLC eye detail points
            pts_in_patch = 0
            for i in range(4, num_pts):
                x, y, lik = dlc_face.values[index, i*3:i*3+3]
                if lik >= dlc_conf_thresh:
                    xr, yr = x - x0, y - y0
                    if 0 <= xr < eye_w and 0 <= yr < eye_h: # prevents plotting points that lie outside the 240×180 crop
                        pts_in_patch += 1
                        axd['upper right'].scatter(xr, yr, alpha=0.8, s=100, color='w')

            # boundary check: avoids ValueError if the line end fell out of crop
            if (0 <= dlc_face.values[index][12] - x0 < eye_w
                and 0 <= dlc_face.values[index][13] - y0 < eye_h
                and 0 <= dlc_face.values[index][15] - x0 < eye_w
                and 0 <= dlc_face.values[index][16] - y0 < eye_h):
                axd['upper right'].plot(
                    [dlc_face.values[index][12] - x0, dlc_face.values[index][15] - x0],   
                    [dlc_face.values[index][13] - y0, dlc_face.values[index][16] - y0],
                    color='yellow', linewidth=1.5
                )
            if (0 <= dlc_face.values[index][12] - x0 < eye_w
                and 0 <= dlc_face.values[index][13] - y0 < eye_h
                and 0 <= dlc_face.values[index][18] - x0 < eye_w
                and 0 <= dlc_face.values[index][19] - y0 < eye_h):
                axd['upper right'].plot(
                    [dlc_face.values[index][12] - x0, dlc_face.values[index][18] - x0],  
                    [dlc_face.values[index][13] - y0, dlc_face.values[index][19] - y0],
                    color='yellow', linewidth=1.5
                )

            # count in‐patch DLC pupil points
            pupil_in_patch = 0
            for ci in pupil_pts:
                x, y, lik = dlc_face.values[index, ci*3:ci*3+3]
                if lik >= dlc_conf_thresh:
                    xr, yr = x - x0, y - y0
                    if 0 <= xr < eye_w and 0 <= yr < eye_h:
                        pupil_in_patch += 1
            # Draw circle
            cx = face.loc[index, ('smooth','PupilX')] - x0
            cy = face.loc[index, ('smooth','PupilY')] - y0
            r  = face.loc[index, ('smooth','PupilDiam')]
            if pupil_in_patch >= 3 and (cx - r >= 0) and (cy - r >= 0) and (cx + r <= eye_w) and (cy + r <= eye_h):
                circle = plt.Circle((cx, cy), r, edgecolor=cmap(0), facecolor='none', linewidth=3, alpha=0.8)
                axd['upper right'].add_patch(circle)

            plt.tight_layout()
            plt.savefig(os.path.join(resultsdir, f"file_{index:06d}.png"))

            pbar.update(1)
            index += 1
            if index >= lastframe:
                break
            previous_frame = current_frame # update to the frame just finished processing, so at the start of the next iteration it truly holds the previous frame
            ret, current_frame = facemp4.read()
            if current_frame is None:
                break
    facemp4.release()
    if make_video:
        frames = sorted(glob.glob(os.path.join(resultsdir, 'file_*.png')))
        if not frames:
            raise RuntimeError(f"No frames found in {resultsdir}. Ensure the save pattern matches the glob.")
        vid_dirout = os.path.join(resultsdir, 'labeled_face_video.mp4')
        first = cv2.imread(frames[0])
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(vid_dirout, fourcc, FaceCam_FPS, (w, h))
        for fname in frames:
            img = cv2.imread(fname)
            writer.write(img)
        writer.release()
        print(f"Saved labeled video to {vid_dirout}")



def create_labeled_video_body(params, body, write_mp4=False):
    dlc_bodyvid = cv2.VideoCapture(glob.glob(os.path.join(params['paths']['Results_DLC_Body'], '') + '*DeepCut*.mp4')[0])
    if params['cameras']['body_output'] > 1:
        vidlength = round(params['cameras']['body_output'])
    else:
        vidlength = int(dlc_bodyvid.get(7))
    print("Labelling body video...")
    ret, current_frame = dlc_bodyvid.read()
    index = 0

    plt.close("all")
    plt.ioff()  # hide figures
    # plt.ion()  # show figures
    fig = plt.figure(figsize=(10, 8), dpi=80)

    if write_mp4:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter(params['paths']['Results_BodyVid'], fourcc, int(params['cameras']['BodyCam_FPS']), (800, 640))
    elif not os.path.exists(os.path.join(params['paths']['Results_Cam_Dir'], 'body_output_vid')):
        os.makedirs(os.path.join(params['paths']['Results_Cam_Dir'], 'body_output_vid'))

    with tqdm(total=vidlength) as pbar:
        while dlc_bodyvid.isOpened():
            # shift X window of data
            plt.subplot(212)
            plt.cla()
            bodynames = list(body.columns)
            bodydatapoints = (body.values[:] - np.nanmean(np.array(body.values, dtype=float), axis=0)) / np.sqrt(np.nanstd(np.array(body.values, dtype=float), axis=0))
            plt.xlim(0, 750)
            plt.ylim(-25, 50)
            bodydatapointsshifted = shift5(bodydatapoints, 375-index)
            for i in range(len(bodydatapoints.transpose())):
                plt.plot(bodydatapointsshifted.transpose()[i] + i)
                plt.legend(bodynames, loc=2)
            plt.axvline(x=375, color='black')

            # plot body frame
            plt.subplot(211)
            plt.cla()
            plt.imshow(current_frame)
            plt.axis('off')
            plt.tight_layout()
            plt.text(10, 30, str(params['paths']['Data_BodyCam'][-22:]), color='w', size=8)
            plt.text(10, 60, "%06d" % index, color='w')

            if write_mp4:
                out.write(mplfig_to_npimage(fig))
            else:
                plt.savefig(os.path.join(params['paths']['Results_Cam_Dir'], 'body_output_vid', '') + "file%06d.png" % index)

            pbar.update(1)
            index = index+1
            if index > vidlength:
                break
            ret, current_frame = dlc_bodyvid.read()
            if current_frame is None:
                break
    if write_mp4:
       out.release()
    dlc_bodyvid.release()

def plot_sweep_inference(params, which_camera, triggersum, inferredtriggersum, total_frame_diff_diff):
    plt.figure(num=None, figsize=(16, 12), dpi=80, facecolor='w', edgecolor='k')
    plotrows = round(len(triggersum) / 5) + 2
    plotcols = 5
    plt.subplot(plotrows,1,1)
    dashsize = int(total_frame_diff_diff.max()/10)
    for t in triggersum:
        plt.axvline(t, dashes=(dashsize, dashsize), color='r')
    for u in inferredtriggersum:
        plt.axvline(u, color='g')
    total_frame_diff_diff.plot(color='b')
    plt.axis('off')
    plt.title('Recorded (red) and inferred (green) sweep transitions for {} camera; max sweep inference {} frames'.format(which_camera, params['cameras']['infer_sweep_maxframes']))
    for idx in range(len(triggersum)-1):
        triggerrange = np.arange(inferredtriggersum[idx]-params['cameras']['infer_sweep_maxframes'], inferredtriggersum[idx]+params['cameras']['infer_sweep_maxframes'])
        plt.subplot(plotrows, plotcols, idx+6)
        plt.axvline(triggersum[idx], dashes=(dashsize, dashsize), color='r')
        plt.axvline(inferredtriggersum[idx], color='g')
        total_frame_diff_diff[triggerrange].plot(color='b')
        plt.axis('off')
        plt.title('{}: Diff: {} frames'.format(idx, triggersum[idx]-inferredtriggersum[idx]))
        plt.ylim(total_frame_diff_diff[triggerrange].min()*2, total_frame_diff_diff[triggerrange].max()*2)
    plt.savefig(os.path.join(params['paths']['Results_Cam_Dir'], '') + which_camera + "_sweep_inference.pdf")
    plt.close('all')

