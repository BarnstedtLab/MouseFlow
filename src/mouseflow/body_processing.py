"""
Author: @Jakob Faust
Date: 10.10.2026
"""


import pandas as pd
import numpy as np
from scipy.stats.mstats import zscore

from typing import NamedTuple

from mouseflow.utils.motion_processing import smooth, freq_analysis2
from mouseflow.utils.preprocess_video import get_video_for_marker, video_info
from mouseflow.utils.handling_markers import load_markers

from pathlib import Path

class AngleResult(NamedTuple):
    angle3: pd.Series
    slope: pd.Series

class PointMotionResult(NamedTuple):
    raw_distance: pd.Series
    proc_distance: pd.Series
    angles: pd.Series

class BodyAnalyzer():
    def __init__(self):
        pass

    def _as_series(self, x, name, idx):
        x = pd.Series(x, name=name)
        return x.reindex(idx)

    def _dlc_pointdistance(self, point1, point2, body_conf_thresh=.5, body_interpolation_limit=300):
        point1, point2 = point1.values, point2.values
        confident = [(point1[:, 2] > body_conf_thresh), (point2[:, 2] > body_conf_thresh)]
        point1_conf = pd.DataFrame(np.array((point1[:,0]*confident[0], point1[:,1]*confident[0])).T, columns = ['x', 'y'])
        point1_conf[point1_conf == 0] = np.nan
        point2_conf = pd.DataFrame(np.array((point2[:,0]*confident[1], point2[:,1]*confident[1])).T, columns = ['x', 'y'])
        point2_conf[point2_conf == 0] = np.nan
        xydiff_raw = point1_conf - point2_conf
        xydist_raw = pd.Series(np.linalg.norm(xydiff_raw, axis=1))
        xydist_interp = xydist_raw.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        xydist_z = ((xydist_interp - xydist_interp.mean()) / xydist_interp.std(ddof=0))[:len(point1)]

        return pd.Series(xydist_raw), pd.Series(xydist_z)

    def _dlc_angle(self, point1, point2, point3, body_conf_thresh=.5, body_smooth_window=25, body_interpolation_limit=150) -> AngleResult:
        point1, point2, point3 = point1.values, point2.values, point3.values
        confident = [(point1[:,2] > body_conf_thresh), (point2[:,2] > body_conf_thresh), (point3[:,2] > body_conf_thresh)]
        point1_conf = pd.DataFrame(np.array((point1[:,0]*confident[0], point1[:,1]*confident[0])).T, columns = ['x', 'y'])
        point1_conf[point1_conf == 0] = np.nan
        point2_conf = pd.DataFrame(np.array((point2[:,0]*confident[1], point2[:,1]*confident[1])).T, columns = ['x', 'y'])
        point2_conf[point2_conf == 0] = np.nan
        point3_conf = pd.DataFrame(np.array((point3[:,0]*confident[2], point3[:,1]*confident[2])).T, columns = ['x', 'y'])
        point3_conf[point3_conf == 0] = np.nan
        dX1 = np.array(point2_conf['x'] - point1_conf['x'], dtype=float)
        dY1 = np.array(point2_conf['y'] - point1_conf['y'], dtype=float)
        deg1 = pd.Series(np.arctan2(dY1, dX1)) + np.pi
        deg1.loc[deg1>np.pi] = deg1.loc[deg1>np.pi] - 2*np.pi
        dX2 = np.array(point3_conf['x'] - point2_conf['x'], dtype=float)
        dY2 = np.array(point3_conf['y'] - point2_conf['y'], dtype=float)
        deg2 = pd.Series(np.arctan2(dY2, dX2))  + np.pi
        deg2.loc[deg2>np.pi] = deg2.loc[deg2>np.pi] - 2*np.pi
        angle = deg2 - deg1
        deg1_interp = deg1.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        angle_interp = angle.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        deg1_smooth = pd.Series(smooth(deg1_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
        angle_smooth = pd.Series(smooth(angle_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
        return AngleResult(angle3=pd.Series(angle_smooth), slope=pd.Series(deg1_smooth))


    def _dlc_pointmotion(self, dlc, body_conf_thresh=.5, body_smooth_window=25, body_interpolation_limit=150) -> PointMotionResult:
        x = dlc['x']
        y = dlc['y']
        likelihood = dlc['likelihood']
        confident = likelihood > body_conf_thresh
        x_conf = x*confident
        y_conf = y*confident
        xy = pd.DataFrame(np.array((x_conf, y_conf)).T, columns=['x', 'y'])
        xy[xy == 0] = np.nan
        xydiff_raw = xy.diff()
        xydist_raw = pd.Series(np.linalg.norm(xydiff_raw, axis=1))
        deg = pd.Series(np.arctan2(xydiff_raw.y, xydiff_raw.x)) + np.pi
        deg.loc[deg>np.pi] = deg.loc[deg>np.pi] - 2*np.pi
        xydist_interp = xydist_raw.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        deg_interp = deg.interpolate(method='linear', limit=body_interpolation_limit)  # linear interpolation
        xydist_z_raw = ((xydist_interp - xydist_interp.mean()) / xydist_interp.std(ddof=0))[:len(x)]
        xydist_smooth = pd.Series(smooth(xydist_interp, window_len=body_smooth_window)).shift(periods=-int(body_smooth_window/2))
        xydist_z = ((xydist_smooth - xydist_smooth.mean()) / xydist_smooth.std(ddof=0))[:len(x)]

        return PointMotionResult(pd.Series(xydist_raw), pd.Series(xydist_z), pd.Series(deg_interp))

    def analyse(self, body_file: Path, keypoint_dir: Path):
        out_file = body_file.with_name(body_file.stem + '_mouseflow.h5')
        if out_file.exists() and not self.cfg.overwrite:
            print(f"{out_file} already analysed, skipping ahead...")
            return
        
        markers_body = load_markers(body_file, key=None)
        video_file = get_video_for_marker(body_file, keypoint_dir)
        fps, w, h, cap = video_info(video_file)

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
        motion_frontpaw = self._dlc_pointmotion(dlc_paw_front)
        motion_backpaw = self._dlc_pointmotion(dlc_paw_back)

        # Paw angles
        angle_paws_front = self._dlc_angle(
            markers_body['paw_front-right1'], markers_body['paw_front-right2'], markers_body['paw_front-right3'])
        angle_paws_back = self._dlc_angle(
            markers_body['paw_back-right1'],  markers_body['paw_back-right2'],  markers_body['paw_back-right3'])

        # Stride and gait information
        rightpaws_fbdiff_raw, _  = self._dlc_pointdistance(
            markers_body['paw_front-right2'], markers_body['paw_back-right2'])
        stride_freq = freq_analysis2(rightpaws_fbdiff_raw, fps)

        # Mouth motion
        dlc_mouth = (
            markers_body
            .loc[:, ('mouth', ['x','y','likelihood'])]  # select the 3 subcols
            .droplevel(0, axis=1)                                   # leave cols: x, y, likelihood
        )
        motion_mouth = self._dlc_pointmotion(dlc_mouth)

        # Tail information
        angle_tail = self._dlc_angle(
            markers_body['tail1'], markers_body['tail2'], markers_body['tail3'])

        idx = markers_body.index
        

        # Due to smoothing we might get different lebgths compared to the raw data.
        # Therefore, we just fill up the shorter rows with NaNs to match the raw data lengths. 
        body_raw = pd.concat([
            self._as_series(motion_frontpaw.raw_distance, 'PointMotion_FrontPaw', idx),
            self._as_series(motion_frontpaw.angles,       'AngleMotion_FrontPaw', idx),
            self._as_series(motion_mouth.raw_distance,    'PointMotion_Mouth', idx),
            self._as_series(motion_mouth.angles,          'AngleMotion_Mouth', idx),
            self._as_series(motion_backpaw.raw_distance,  'PointMotion_BackPaw', idx),
            self._as_series(motion_backpaw.angles,        'AngleMotion_BackPaw', idx),
            self._as_series(stride_freq, 'Frequency_BackPaw', idx),
            self._as_series(angle_tail.angle3,            'Angle_Tail_3', idx),
            self._as_series(angle_tail.slope,             'Angle_Tail', idx),
            self._as_series(angle_paws_front.angle3,      'Angle_Paws_Front_3', idx),
            self._as_series(angle_paws_front.slope,       'Angle_Paws_Front', idx),
            self._as_series(angle_paws_back.angle3,       'Angle_Paws_Back_3', idx),
            self._as_series(angle_paws_back.slope,        'Angle_Paws_Back', idx),
            self._as_series(-zscore(markers_body['tail1','y']), 'Tailroot_Level', idx),
        ], axis=1)


        # further process raw data and save
        cap.release()
        body_raw.to_hdf(out_file, key='body')
        print(f"saved mouseflow body analysis to {out_file}")
