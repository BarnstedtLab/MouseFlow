'''
Author: @Jakob Faust
Date: 26.03.2026

Optical Flow class that abstracts
    1) GPU: OpenCV Farneback Optical Flow (https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) with cuda support
    2) CPU: OpenCV DIS Optical flow (https://doi.org/10.48550/arXiv.1603.03590, https://docs.opencv.org/3.4/da/d06/classcv_1_1optflow_1_1DISOpticalFlow.html)

Both implementations share the same BaseOF interface to allow seamless switching between GPU Farneback and CPU DIS based approaches.
The interface also provides an easy way for integrating other optical flow algorithms into the package. 

The Farneback implementation is optimized to run on a Nvidia GPU with cuda support concentrating on efficient host-device data transfer:
We have two seperate GPU streams communicating via events.
The first stream handles copying to and from GPU storage, the second stream performs actual GPU calculations.
This results in asynchronous and highly performant Optical Flow Calculations. 

The DIS implementation concentrates on vectorizing operations on CPUs. 
'''

import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm
from pathlib import Path


import cv2


class BaseOF(ABC):
    @abstractmethod
    def set_masks(self, masks: list[np.ndarray]):
        """takes face region masks and pre-uploads them to GPU to reduce costly host-device memory transactions"""
        ...
    
    @abstractmethod
    def open(self, video_path: str, start=0, end=None):
        """opens video file und uploads first two frames to GPU (as initial check)"""
        ...
    
    @abstractmethod
    def run(self):
        """Runs optical flow on all frames and returns result as dict of numpy arrays on CPU"""
        ...
    
    @abstractmethod
    def video_info(self):
        """Returns number of video frames"""
        ...

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

class FarnebackOF(BaseOF):
    def __init__(self, config_args=None, n_iters=3):
        super().__init__()
        self.args = config_args or dict(
            numLevels=5, pyrScale=.5, fastPyramids=True, winSize=25,
            numIters=n_iters, polyN=5, polySigma=1.2, flags=0
        )
        self.of = cv2.cuda_FarnebackOpticalFlow.create(**self.args)
        self.compute_stream = cv2.cuda.Stream()
        self.copy_stream = cv2.cuda.Stream()
        
        # State variables describing the whole intput 
        self.masks_np = None
        self.masks_gpu = []
        self.px_per_mask = None
        self.shape = None
        self.video_path = None
        self.cap = None

        # Video info
        self.nframes = 0
        self.fps = 0
        self.height = 0
        self.width = 0

        # Persistent GPU/CPU Buffers needed for tracking processing progress
        self.gpu_bgr = None
        self.gpu_gray = None
        self.gpu_gray_f32 = None
        self.host_buf = None 

        self.save_vectors = True
        self.downsample_factor = 8
        self.ready_evt = cv2.cuda.Event()

    def set_masks(self, masks: list[np.ndarray]):
        '''Uploads fixed ROI masks to the GPU'''
        self.masks_np = [np.ascontiguousarray(m.astype('f4')) for m in masks]
        self.px_per_mask = np.array([m.sum() for m in masks], dtype='f4')
        self.masks_gpu = []
        
        for m in self.masks_np:
            m_gpu = cv2.cuda_GpuMat()
            m_gpu.upload(m, stream=self.copy_stream)
            self.masks_gpu.append(m_gpu)
            
        self.copy_stream.waitForCompletion()
        print(f"{len(self.masks_np)} Masks uploaded to GPU")
    
    def video_info(self):
        return self.nframes, self.fps, self.height, self.width

    def open(self, video_path: str, start=0, end=None):
        '''Opens a video file and primes the first two frames onto the GPU'''
        self.close() 

        self.cap = cv2.VideoCapture(video_path)
        self.video_path = Path(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'Cannot open video {video_path}')
        
        self.start = start
        if self.start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
            
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nframes = total if end is None else min(end, total)

        ret, frame0 = self.cap.read()
        ret, frame1 = self.cap.read() if ret else (False, None)
        if not ret:
            self.close()
            raise RuntimeError("Couldn't read initial frames")
            
        self.shape = frame0.shape
        self.height, self.width = self.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Pre-allocate pinned memory and GPU buffers for the two rotating frames
        self.host_buf = np.empty((self.height, self.width, 3), dtype=np.uint8)
        cv2.cuda.registerPageLocked(self.host_buf) 

        self.gpu_bgr = [cv2.cuda_GpuMat(self.height, self.width, cv2.CV_8UC3) for _ in range(2)]
        self.gpu_gray = [cv2.cuda_GpuMat(self.height, self.width, cv2.CV_8UC1) for _ in range(2)]
        self.gpu_gray_f32 = [cv2.cuda_GpuMat(self.height, self.width, cv2.CV_32FC1) for _ in range(2)]

        # Upload and convert the first two frames
        for i, frame in enumerate([frame0, frame1]):
            self.host_buf[:] = frame
            self.gpu_bgr[i].upload(self.host_buf, stream=self.copy_stream)
            cv2.cuda.cvtColor(self.gpu_bgr[i], cv2.COLOR_BGR2GRAY, dst=self.gpu_gray[i], stream=self.copy_stream)
            self.gpu_gray_f32[i] = self.gpu_gray[i].convertTo(cv2.CV_32F, stream=self.copy_stream)

        self.ready_evt.record(self.copy_stream)
        self.compute_stream.waitEvent(self.ready_evt)

    def close(self):
        if hasattr(self, 'host_buf') and self.host_buf is not None:
            try: cv2.cuda.unregisterPageLocked(self.host_buf)
            except Exception: pass
            self.host_buf = None
            
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None

    def __del__(self):
        self.close()

    def _allocate_run_buffers(self, max_frames, num_masks, ds_height, ds_width):
        """
        Helper function to allocate GPU and CPU RAM memory efficiently
        The function is needed for the optical flow calculations 
        """

        # GPU arrays for optical flow calculations and roi wise averaging
        gpu = {
            'flow': cv2.cuda_GpuMat(self.height, self.width, cv2.CV_32FC2),
            'flow_x': cv2.cuda_GpuMat(self.height, self.width, cv2.CV_32FC1),
            'flow_y': cv2.cuda_GpuMat(self.height, self.width, cv2.CV_32FC1),
            'flow_small': cv2.cuda_GpuMat(ds_height, ds_width, cv2.CV_32FC2),
            'flow_x_small': cv2.cuda_GpuMat(ds_height, ds_width, cv2.CV_32FC1),
            'flow_y_small': cv2.cuda_GpuMat(ds_height, ds_width, cv2.CV_32FC1),
            
            # ROI intermediate buffers
            'sum_u64': [cv2.cuda_GpuMat(1, 1, cv2.CV_64F) for _ in range(num_masks)],
            'sum_v64': [cv2.cuda_GpuMat(1, 1, cv2.CV_64F) for _ in range(num_masks)],
            'sum_diff64': [cv2.cuda_GpuMat(1, 1, cv2.CV_64F) for _ in range(num_masks)],
            'sum_u32': [cv2.cuda_GpuMat(1, 1, cv2.CV_32F) for _ in range(num_masks)],
            'sum_v32': [cv2.cuda_GpuMat(1, 1, cv2.CV_32F) for _ in range(num_masks)],
            'sum_diff32': [cv2.cuda_GpuMat(1, 1, cv2.CV_32F) for _ in range(num_masks)],
        }
        
        # CPU arrays for output handling
        cpu = {
            'mag': np.empty((max_frames, num_masks), dtype=np.float32),
            'ang': np.empty((max_frames, num_masks), dtype=np.float32),
            'diff': np.empty((max_frames, num_masks), dtype=np.float32),
            'flow_grid': np.ascontiguousarray(np.empty((max_frames, ds_height, ds_width, 2), dtype=np.float32))
        }
        
        # Keep all cpu arrays in RAM at fixed positions for faster GPU - CPU data transfre
        cv2.cuda.registerPageLocked(cpu['mag'])
        cv2.cuda.registerPageLocked(cpu['ang'])
        cv2.cuda.registerPageLocked(cpu['diff'])
        
        return gpu, cpu

    def run(self):
        max_frames = max(0, self.nframes - (self.start + 1))
        num_masks = len(self.masks_gpu)
        ds_height = self.height // self.downsample_factor
        ds_width = self.width // self.downsample_factor

        gpu, cpu = self._allocate_run_buffers(max_frames, num_masks, ds_height, ds_width)
        compute_done_evt = cv2.cuda.Event()

        frame_idx = self.start + 1
        out_idx = 0
        cur, prev = 1, 0

        try:
            with tqdm(total=max_frames) as pbar:
                while True:
                    # Compute optical flow and downsample for later flow field saving
                    self.of.calc(self.gpu_gray[prev], self.gpu_gray[cur], gpu['flow'], stream=self.compute_stream)
                    cv2.cuda.split(gpu['flow'], [gpu['flow_x'], gpu['flow_y']], stream=self.compute_stream)
                    
                    cv2.cuda.resize(gpu['flow_x'], (ds_width, ds_height), dst=gpu['flow_x_small'], interpolation=cv2.INTER_AREA, stream=self.compute_stream)
                    cv2.cuda.resize(gpu['flow_y'], (ds_width, ds_height), dst=gpu['flow_y_small'], interpolation=cv2.INTER_AREA, stream=self.compute_stream)
                    cv2.cuda.merge([gpu['flow_x_small'], gpu['flow_y_small']], gpu['flow_small'], stream=self.compute_stream)
                    
                    diff = cv2.cuda.absdiff(self.gpu_gray_f32[cur], self.gpu_gray_f32[prev], stream=self.compute_stream)
                    compute_done_evt.record(self.compute_stream)

                    # Asynchronously read and upload next video frame from CPU to GPU
                    ret, next_frame = self.cap.read()
                    if not ret:
                        break
                        
                    self.host_buf[:] = next_frame 
                    self.gpu_bgr[prev].upload(self.host_buf, stream=self.copy_stream) 
                    cv2.cuda.cvtColor(self.gpu_bgr[prev], cv2.COLOR_BGR2GRAY, dst=self.gpu_gray[prev], stream=self.copy_stream)
                    self.gpu_gray_f32[prev] = self.gpu_gray[prev].convertTo(cv2.CV_32F, stream=self.copy_stream)

                    # Download downsampled flow to cpu (while processing continous on gpu) 
                    self.copy_stream.waitEvent(compute_done_evt)
                    gpu['flow_small'].download(stream=self.copy_stream, dst=cpu['flow_grid'][out_idx])
                    
                    # ROI wise statistics
                    for i, (mask_gpu, px_count) in enumerate(zip(self.masks_gpu, self.px_per_mask)):
                        # Mask out all pixels that are not within a ROI
                        flow_x_masked = cv2.cuda.multiply(gpu['flow_x'], mask_gpu, stream=self.compute_stream)
                        flow_y_masked = cv2.cuda.multiply(gpu['flow_y'], mask_gpu, stream=self.compute_stream)
                        diff_masked = cv2.cuda.multiply(diff, mask_gpu, stream=self.compute_stream)

                        # Sum all vectors ROI wise and get the absolute difference ROI wise
                        cv2.cuda.calcSum(flow_x_masked, gpu['sum_u64'][i], mask=None, stream=self.compute_stream)
                        cv2.cuda.calcSum(flow_y_masked, gpu['sum_v64'][i], mask=None, stream=self.compute_stream)
                        cv2.cuda.calcAbsSum(diff_masked, gpu['sum_diff64'][i], mask=None, stream=self.compute_stream)

                        # Downcast averages from float64 to float32 and download to CPU RAM 
                        gpu['sum_u32'][i] = gpu['sum_u64'][i].convertTo(cv2.CV_32F, stream=self.compute_stream)
                        gpu['sum_v32'][i] = gpu['sum_v64'][i].convertTo(cv2.CV_32F, stream=self.compute_stream)
                        gpu['sum_diff32'][i] = gpu['sum_diff64'][i].convertTo(cv2.CV_32F, stream=self.compute_stream)

                        u = gpu['sum_u32'][i].download(stream=self.copy_stream)
                        v = gpu['sum_v32'][i].download(stream=self.copy_stream)
                        cpu['diff'][out_idx:out_idx+1, i] = gpu['sum_diff32'][i].download(stream=self.copy_stream)

                        # Calculate mean magnitude and angle on cpu
                        mean_u = u / px_count
                        mean_v = v / px_count
                        cpu['mag'][out_idx, i] = np.sqrt(mean_u**2 + mean_v**2)
                        cpu['ang'][out_idx, i] = np.arctan2(mean_v, mean_u)

                    #cleanup and rotate bufferfs to process next frame
                    pbar.update(1)
                    frame_idx += 1
                    out_idx += 1
                    if frame_idx >= self.nframes:
                        break

                    self.ready_evt.record(self.copy_stream)
                    self.compute_stream.waitEvent(self.ready_evt)

                    # Swap buffers
                    prev, cur = cur, prev 

            # Format final outputs
            px = np.maximum(np.asarray(self.px_per_mask, dtype=np.float32), 1.0)
            return dict(
                diff = (cpu['diff'][:out_idx, :] / px[None, :]).astype(np.float32),
                mag  = cpu['mag'][:out_idx, :].astype(np.float32),
                ang  = cpu['ang'][:out_idx, :].astype(np.float32),
                flow_grid = cpu['flow_grid'][:out_idx].transpose(0, 3, 1, 2)
            )

        finally:
            self.copy_stream.waitForCompletion()
            self.compute_stream.waitForCompletion()
            cv2.cuda.unregisterPageLocked(cpu['mag'])
            cv2.cuda.unregisterPageLocked(cpu['ang'])
            cv2.cuda.unregisterPageLocked(cpu['diff'])
            self.close()


class DISFlowOF:
    def __init__(self, preset=cv2.DISOPTICAL_FLOW_PRESET_MEDIUM):
        """
        DIS Optical Flow.
        """
        cv2.setNumThreads(8)
        
        self.dis = cv2.DISOpticalFlow_create(preset)
        # self.dis.setPatchStride(3)
        
        # self.dis.setFinestScale(1)

        self.masks = []
        self.px_per_mask = None
        self.cap = None
        self.start = 0
        self.nframes = 0
        self.fps = 0
        self.height = 0
        self.width = 0
        self.save_vectors = True
        self.downsample_factor = 8
        self.video_path = None
        
        self.prev_gray = None
        self.cur_gray = None
        self.flow = None

    
    def set_masks(self, masks: list[np.ndarray]):
        for i, m in enumerate(masks):
            if m.shape != (self.height, self.width):
                m = m.T
            if m.shape != (self.height, self.width):
                raise ValueError(f"Mask shape {m.shape} mismatch with video {(self.height, self.width)}")
            masks[i] = m
            
        self.masks = [m.astype(np.float32) for m in masks]
        self.px_per_mask = np.array([max(m.sum(), 1.0) for m in masks], dtype=np.float32)
        print(f"{len(self.masks)} Masks set for DIS processing")

    def video_info(self):
        return self.nframes, self.fps, self.height, self.width

    def open(self, video_path: str, start=0, end=None):
        self.cap = cv2.VideoCapture(video_path)
        self.video_path = Path(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open video {video_path}')
        
        self.start = start
        if self.start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
            
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nframes = total if end is None else min(end, total)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # Read first frame to init dimensions
        ret, frame0 = self.cap.read()
        if not ret: raise RuntimeError("couldn't read frame 0")
        
        self.height, self.width = frame0.shape[:2]
        
        # Init Flow Buffer (Re-using this array prevents memory fragmentation)
        self.mag  = np.empty((self.height, self.width), np.float32)
        self.ang  = np.empty((self.height, self.width), np.float32)


        # Prepare prev_gray
        self.prev_gray = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        
        # Rewind if we are starting at 0 (since we just read frame 0)
        if self.start == 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    

    def run(self):
        max_frames = max(0, self.nframes - self.start)
        pbar = tqdm(total=max_frames, unit="frames")
        mags_out = np.empty((max_frames, len(self.masks)), dtype=np.float32)
        angs_out = np.empty((max_frames, len(self.masks)), dtype=np.float32)
        diffs_out = np.empty((max_frames, len(self.masks)), dtype=np.float32)

        masks_matrix = np.array(self.masks).reshape(len(self.masks), -1)
        px_counts = self.px_per_mask

        ret, frame = self.cap.read()
        if not ret: return {}
        
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w_downsampled = int(self.width // self.downsample_factor)
        h_downsampled = int(self.height // self.downsample_factor)
        flow_grids = np.zeros((max_frames, h_downsampled, w_downsampled, 2), dtype=np.float32)
        self.flow = np.empty((self.height, self.width, 2), dtype=np.float32)
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret or frame_idx >= max_frames - 1:
                break
            
            self.cur_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.flow = self.dis.calc(self.prev_gray, self.cur_gray, self.flow)

            flow = self.flow.reshape(-1, 2)
            flow_u = flow[:, 0]
            flow_v = flow[:, 1]
            mean_x = (masks_matrix @ flow_u) / px_counts
            mean_y = (masks_matrix @ flow_v) / px_counts

            mags_out[frame_idx] = np.sqrt(mean_x**2 + mean_y**2)
            angs_out[frame_idx] = np.arctan2(mean_y, mean_x)


            diff = cv2.absdiff(self.cur_gray, self.prev_gray).astype(np.float32)
            diffs_out[frame_idx] = (masks_matrix @ diff.reshape(-1)) / px_counts

            if self.save_vectors:
                flow_grids[frame_idx] = cv2.resize(
                    self.flow, 
                    (w_downsampled, h_downsampled), 
                    interpolation=cv2.INTER_AREA
                )

            self.prev_gray = self.cur_gray
            pbar.update(1)
            frame_idx += 1

        self.cap.release()
        pbar.close()

    
        return dict(
            diff=diffs_out[:frame_idx],
            mag=mags_out[:frame_idx],
            ang=angs_out[:frame_idx],
            flow_grid=flow_grids[:frame_idx].transpose(0, 3, 1, 2)
        )
