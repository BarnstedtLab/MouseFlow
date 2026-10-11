"""
Author: @Jakob Faust
Date: 10.10.2026
"""

from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import math
import h5py
from dataclasses import dataclass


from mouseflow.utils import confidence_na, load_markers
from mouseflow.optical_flow import BaseOF, FarnebackOF, DISFlowOF
from mouseflow.utils.preprocess_video import get_video_for_marker, video_info
from mouseflow.utils.motion_processing import smooth, freq_analysis2, dlc_pointdistance2, hilbert_peaks

@dataclass(frozen=True)
class FaceConfig:
    conf_thresh: float | None = None
    interpolation_limits_sec: dict = ...
    smoothing_windows_sec: dict = ...
    na_limit: float = 0.25
    faceregions_sizes: dict | None = None
    base_resolution: tuple[int,int] | None = None
    manual_anchor: dict | None = None

class FaceAnalyzer():
    def __init__(self, cfg:FaceConfig=None, opticalflow_enigne:str='DIS', save_vectorfield: bool=False):
        self.opticalflow_enigne = opticalflow_enigne
        self.save_vectorfield = save_vectorfield

        if cfg:
            self.cfg = cfg
        else:
            conf_thresh = None
            interpolation_limits_sec={
                'pupil': 2,
                'eyelid': 1,
            }
            smoothing_windows_sec={
                'PupilDiam': 1,
                'PupilMotion': 0.25,
                'eyelid': 0.1,
                'MotionEnergy': 0.25,
            }
            na_limit=0.25
            faceregions_sizes=None
            base_resolution=None
            manual_anchor=None
            self.cfg = FaceConfig(conf_thresh, interpolation_limits_sec,
                       smoothing_windows_sec, na_limit, faceregions_sizes,
                       base_resolution, manual_anchor)

    def _process_raw_data(self, smoothing_windows_sec, na_limit, FaceCam_FPS, interpolation_limits_frames, face_raw):
        face_raw.iloc[:, (face_raw.isnull().mean()>na_limit).values] = np.nan  

        # Interpolate missing values
        face_interp = face_raw.copy()
        face_interp[['PupilX', 'PupilY', 'PupilMotion', 'PupilDiam']] = \
                face_interp[['PupilX', 'PupilY', 'PupilMotion', 'PupilDiam']].interpolate(
                    method='linear', limit=interpolation_limits_frames['pupil'])

        # Smoothen data
        smoothing_windows_frames = {x: int(k * FaceCam_FPS)
                                    for (x, k) in smoothing_windows_sec.items()}
        face_smooth = face_interp.copy()
        face_smooth['PupilDiam'] = face_smooth['PupilDiam'].rolling(
                window=smoothing_windows_frames['PupilDiam'], center=True).mean().astype('float32')
        face_smooth[['PupilX', 'PupilY', 'PupilMotion']] = \
                face_smooth[['PupilX', 'PupilY', 'PupilMotion']].rolling(
                window=smoothing_windows_frames['PupilMotion'], center=True).mean().astype('float32')
        face_smooth.loc[:, face_smooth.columns.str.startswith('MotionEnergy')] = \
                face_smooth.loc[:, face_smooth.columns.str.startswith('MotionEnergy')].rolling(
                window=smoothing_windows_frames['MotionEnergy'], center=True).mean().astype('float32')

        # Z-scoring data
        face_zscore = face_smooth.apply(lambda a: (a - a.mean())/a.std(ddof=0))

        # adding binary data
        face_zscore['Saccades'] = pd.Series(face_smooth['PupilX'].diff().abs() > 1.5).astype(int)
        face_zscore['EyeBlink'] = pd.Series(face_interp['EyeLidDist'] < face_interp['EyeLidDist'].median()*.75).astype(int)

        # Concatenating all data types into multi-level dataframe
        face = pd.concat({'raw': face_raw, 'interpolated': face_interp, 'smooth': face_smooth, 'zscore': face_zscore}, names=['Data_type'], axis=1)

        return face

    def _pupilextraction(self, pupil_markers_xy_confident):
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

    def _eyeblink(self, dlc_face, eyelid_conf_thresh=.999, eyelid_smooth_window=25, eyelid_interpolation_limit=150):
        eyelid_labels_top = np.array(dlc_face.values[:, [15, 16]]) * (dlc_face.values[:, [17, 17]] > eyelid_conf_thresh)
        eyelid_labels_bot = np.array(dlc_face.values[:, [18, 19]]) * (dlc_face.values[:, [20, 20]] > eyelid_conf_thresh)
        eyelid_labels_top[eyelid_labels_top == 0] = np.nan
        eyelid_labels_bot[eyelid_labels_bot == 0] = np.nan
        eyelid_dist_raw = pd.Series(np.linalg.norm(eyelid_labels_top - eyelid_labels_bot, axis=1))
        eyelid_dist_interp = eyelid_dist_raw.interpolate(method='linear', limit=eyelid_interpolation_limit)  # linear interpolation, 2sec maximum
        eyelid_dist_smooth = pd.Series(smooth(eyelid_dist_interp, window_len=eyelid_smooth_window)).shift(periods=-int(eyelid_smooth_window/2))
        eyelid_dist_z = (eyelid_dist_smooth - eyelid_dist_smooth.mean()) / eyelid_dist_smooth.std(ddof=0)
        eyeblinks = eyelid_dist_interp < eyelid_dist_interp.median()*.75

        return eyelid_dist_z, eyeblinks

    # FACE REGIONS
    def _define_faceregions(self, dlc_face, facevid, dlc_file, manual_anchor=None, faceregions_sizes=None, base_resolution=(782, 582)):
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
        if base_resolution is None:
            base_resolution = (H, W)
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
        face_anchor.at['x', 'whiskerpad_center'] = np.nan
        face_anchor.at['y', 'whiskerpad_center'] = np.nan

        # whisker inference
        whisker_r = 0.0
        if nose_pt is None or mouth_pt is None or tear_pt is None:
            mask_whiskers = canvas.astype(bool)
        else:
            centre_whiskers = tuple(np.round(np.vstack([
                nose_pt,
                mouth_pt,
                tear_pt]).mean(axis=0)).astype(int))
            face_anchor.at['x', 'whiskerpad_center'] = centre_whiskers[0]
            face_anchor.at['y', 'whiskerpad_center'] = centre_whiskers[1]
            a = np.linalg.norm(mouth_pt - tear_pt)
            b = np.linalg.norm(tear_pt  - nose_pt)
            c = np.linalg.norm(nose_pt  - mouth_pt)
            s = 0.5*(a+b+c)
            whisker_r = math.sqrt(s*(s-a)*(s-b)*(s-c)) / s
            axes = get_axes('whiskers') or (whisker_r * 1.1, whisker_r * 1.1)
            mask_whiskers = create_mask(centre_whiskers, axes, angle=0)
        # nose inference
        if nose_pt is None:
            mask_nose = canvas.copy()
        else:
            centre_nose = tuple((nose_pt + np.array([0.24 * nose_pt[0], 0.12 * nose_pt[1]])).round().astype(int))
            axes = get_axes('nose') or (whisker_r*2/3, whisker_r*1/2)
            mask_nose = create_mask(centre_nose, axes, angle=-60.0)
        # mouth inference ellipse
        if mouth_pt is None or chin_pt is None:
            mask_mouth = canvas.astype(bool)
        else:
            centre_mouth = tuple(np.round(mouth_pt + (chin_pt - mouth_pt)/3).astype(int))
            angle_mouth = math.degrees(math.atan2(chin_pt[1]  - mouth_pt[1], chin_pt[0]  - mouth_pt[0]))
            dist = np.hypot(chin_pt[0]  - mouth_pt[0], chin_pt[1]  - mouth_pt[1])
            axes = get_axes('mouth') or (dist / 1.75, dist / 4)
            mask_mouth = create_mask(centre_mouth, axes, angle_mouth)
        # cheek inference ellipse
        if chin_pt is None or eyelid_pt is None:
            mask_cheek = canvas.astype(bool)
        else:
            centre_cheek = tuple(np.round(eyelid_pt + (chin_pt - eyelid_pt)/2).astype(int))
            dist = np.linalg.norm(chin_pt - eyelid_pt) 
            axes = get_axes('cheek') or ((2.5/5)*dist, (1/4)*dist)
            mask_cheek = create_mask(centre_cheek, axes, angle=0)
        masks = [mask_nose, mask_whiskers, mask_mouth, mask_cheek]
        if dlc_file:
            plt.imshow(firstframe, cmap='gray')
            plt.savefig(str(dlc_file)[:-3] + "_face_regions.png")
        plt.close('all')
        return masks, face_anchor

    def _has_gpu_support(self):
        '''
        Tests if the installation has proper GPU support (and if a GPU is actually available).
        First check: OpenCV with CUDA support (needs to be compiled on machine!!)
        Second check: PyTorch with CUDA support
        
        if 'Farneback' is specified as of_backend and there is no opencv CUDA installation,
        this will still return False even if it runs on a GPU system!!
        '''
        has_cv2_cuda = hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0

        if not has_cv2_cuda and (self.opticalflow_enigne == "FB"):
            print("No Opencv with GPU support detected, sorry. Ensure to have a working nvidia GPU." \
                "If you have, try switching mf_config.of_backend to 'RAFT' ")
            return False

        if has_cv2_cuda and (self.opticalflow_enigne == "FB"):
            return True

        return False # safe fallback

    def _facemotion(self, videopath, masks, face_anchor, videoslice=[]):
        gpu_flow : BaseOF = None
        if self.opticalflow_enigne == 'FB':
            gpu_flow = FarnebackOF()
        elif self.opticalflow_enigne == 'DIS':
            gpu_flow = DISFlowOF()
        else:
            raise RuntimeError(f"Unexpected flow type {self.opticalflow_enigne}, expected 'DIS' or 'FB'.")
        if videoslice:
            start, end = videoslice[0], videoslice[1]
        else:
            start, end = 0, None
        gpu_flow.open(videopath, start=start, end=end)# opens video
        gpu_flow.set_masks(masks) # upload ROI masks to gpu
        gpu_flow.set_anchors(face_anchor) # anchors for NMF
        out = gpu_flow.run()  # dict of CPU arrays containing below arrays
        motion = np.hstack([out['diff'], out['mag'], out['ang']])
        cols = ['MotionEnergy_Nose','MotionEnergy_Whiskerpad','MotionEnergy_Mouth','MotionEnergy_Cheek',
                'OFmag_Nose','OFmag_Whiskerpad','OFmag_Mouth','OFmag_Cheek',
                'OFang_Nose','OFang_Whiskerpad','OFang_Mouth','OFang_Cheek']
        df_statistics = pd.DataFrame(motion, columns=cols)
        flow_grid = out["flow_grid"] if self.save_vectorfield else None
        return df_statistics, flow_grid

    def _to_h5(self, out_file: Path, face_masks, face_anchor: pd.DataFrame, face: pd.DataFrame):
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

    def analyse(self, face_file: Path, landmark_dir: Path):
        out_file = face_file.with_name(face_file.stem + '_mouseflow.h5')
        out_grid = face_file.with_name(face_file.stem + '_optical_flow_grid.npz')
        if out_file.exists() and not self.overwrite:
            print(f"{out_file} already analysed, skipping ahead...")
            return
        markers_face = load_markers(face_file, key=None)
        video_file = get_video_for_marker(face_file, landmark_dir)
        fps, w, h, cap = video_info(video_file)
        markers_face = confidence_na(self.cfg.conf_thresh, markers_face)
        interpolation_limits = self._interpolation_limits(fps)

        markers_face.loc[:, ['pupil'+str(n+1) for n in range(6)]] = \
            markers_face.loc[:, ['pupil'+str(n+1) for n in range(6)]].interpolate(
                method='linear', limit=interpolation_limits['pupil'])

        # Extracting pupil and eyelid data
        pupil_raw = self._pupilextraction(
            markers_face[['pupil'+str(n+1) for n in range(6)]].values)
        eyelid_dist_raw = pd.Series(dlc_pointdistance2(
            markers_face['eyelid1'], markers_face['eyelid2']), name='EyeLidDist')

        # Define and save face regions
        face_masks, face_anchor = self._define_faceregions(
            markers_face, video_file, face_file, manual_anchor=self.cfg.manual_anchor,
            base_resolution=self.cfg.base_resolution, faceregions_sizes=self.cfg.faceregions_sizes
        )

        # Extract motion in face regions
        if not self._has_gpu_support():
            self.opticalflow_enigne = 'DIS'
            print("No CUDA support detected. Processing with DIS optical flow on your CPU...")
        face_motion, flow_grid = self._facemotion(
            video_file, face_masks, face_anchor)
        whisk_freq = freq_analysis2(
            face_motion['OFang_Whiskerpad'], fps, rollwin=fps, min_periods=int(fps*.67))
        sniff_freq = freq_analysis2(
            face_motion['OFang_Nose'],       fps, rollwin=fps, min_periods=int(fps*.67))
        chewenv, chew = hilbert_peaks(
            face_motion['OFang_Mouth'],    fps)
        face_freq = pd.DataFrame(
            {'Whisking_freq': whisk_freq, 'Sniff_freq': sniff_freq, 'Chewing_Envelope': chewenv, 'Chew': chew})
        face_raw = pd.concat([pupil_raw, eyelid_dist_raw, face_motion, face_freq], axis=1)

        cap.release()
        face = self._process_raw_data(self.cfg.smoothing_windows_sec, self.cfg.na_limit, fps, interpolation_limits, face_raw)
        self._to_h5(out_file, face_masks, face_anchor, face)
        if flow_grid is not None:
            np.savez_compressed(out_grid, flow=flow_grid)
