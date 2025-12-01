'''
Author: @Jakob Faust
Date: 23.10.2025

Optical Flow class that abstracts
    1) Opencv Farneback Optical Flow (https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html) with cuda support
    2) Pytorch Optical Flow based on the RAFT (https://docs.pytorch.org/vision/0.12/auto_examples/plot_optical_flow.html) model

Both implementations share the same BaseOF interface to allow seamless switching between Farneback and Pytorch based approaches.
The interface also provides an easy way for integrating other optical flow algorithms into the package. 
Both implementations are optimized to run on a Nvidia GPU with cuda support with optimized host-device data transfer:
We have two seperate GPU streams communicating via events.

The first stream handles copying to and from GPU storage, the second stream performs actual GPU calculations.
This results in asynchronous and highly performant Optical Flow Calculations. 

'''

import numpy as np
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch
import torch.nn.functional as F

import pandas as pd

import time

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

class RaftOF(BaseOF):
    def __init__(self, n_iters=3):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model()
        self.px_per_mask = None
        self.maks_gpu = []
        self.cap = None
        self.start = 0
        self.niters = n_iters

        self.nframes = 0
        self.fps = 0
        self.height = 0
        self.width = 0

        self.save_vectors = True
        self.downsample_factor = 8

        self.copy_stream = torch.cuda.Stream(device=self.device) if self.device == "cuda" else None
        self.comp_stream = torch.cuda.Stream(device=self.device) if self.device == "cuda" else None
        self.dev_buffers = None
    
    def _load_model(self):
        '''
        Loads RAFT model weights into (CPU / GPU) RAM
        '''
        try:
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
            model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(self.device, dtype=torch.float16).eval()
            return model
        except Exception:
            raise RuntimeError("failed to load the RAFT model")
        
    @staticmethod
    def _to_pinned(bgr_np: np.ndarray) -> torch.Tensor:
        '''
        BGR uint8 HWC (OpenCV) -> RGB float32 [1,3,H,W] in pinned memory
        (Fast memory access by pre allocating conitgous memory blocks in CPU RAM)
        '''
        x = torch.from_numpy(bgr_np[:, :, ::-1].copy())  # HWC --> RGB uint8
        x = x.permute(2,0,1).unsqueeze(0).contiguous()   # [1,3,H,W] (Because Pytorch RAFT expects format of (Batches, Channels, Height, Width))
        if torch.cuda.is_available():
            return x.pin_memory()
        return x
    
    @staticmethod
    def _rgb_to_gray(x: torch.Tensor) -> torch.Tensor:
        # x: [1,3,H,W], float32 [0,1]
        return 0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]
    
    def set_masks(self, masks: list[np.ndarray]):
        '''
        Uploads fixed ROI masks (before applying them to the Optical Flow output) to the GPU
        '''
        mask_tensors = [torch.from_numpy(m).to(self.device, dtype=torch.float32) for m in masks] # shape [1, K, H, W]
        self.maks_gpu = torch.stack(mask_tensors)[:, None, :, :]
        self.px_per_mask = self.maks_gpu.sum(dim=(2, 3))[:, 0].clamp_min(1)
        print(f"Masks are stored on the {self.device} during further processing")
    
    def video_info(self):
        '''
        Just returns basic video info (#frames, fps, height, width)
        '''
        return self.nframes, self.fps, self.height, self.width

    def open(self, video_path: str, start=0, end=None):
        '''
        Opens a video file and uploads the first two frames to the GPU
        Blocks until the upload is finished
        '''
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open video {video_path}')
        self.start = start
        if self.start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nframes = total if end is None else min(end, total)

        # get first 2 frames on cpu
        ret, frame0 = self.cap.read()
        if not ret:
            raise RuntimeError("couldn't read frame 0")
        ret, frame1 = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError("couldn't read frame 0")
        self.shape = frame0.shape
        self.height, self.width = self.shape[:2]
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        # pinned host memory for quick transfer to storage
        host = [self._to_pinned(frame0), self._to_pinned(frame1)]  # [2] of [1,3,H,W] uint8
        # prepare device buffers
        self.dev_buffers  = [torch.empty_like(host[0], device=self.device, dtype=torch.uint8),
                torch.empty_like(host[1], device=self.device, dtype=torch.uint8)]
        
        with torch.cuda.stream(self.copy_stream):
            self.dev_buffers[0].copy_(host[0], non_blocking=True)
            self.dev_buffers[1].copy_(host[1], non_blocking=True)
        if self.device == 'cuda':
            torch.cuda.current_stream(self.device).wait_stream(self.copy_stream)
    
    
    def _pad(self, x):
        # x: [1,3,H,W]
        _, _, h, w = x.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h == 0 and pad_w == 0:
            return x, (0, 0)
        # pad = (pad_left, pad_right, pad_top, pad_bottom) for NCHW 2D padding
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")
        return x_pad, (pad_h, pad_w)

    def _undo_pad(self, t, h, w):
        # t: [B,C,H',W'] or [B,H',W']; returns [:,:h,:w]
        return t[..., :h, :w]
    
    def _downsample_flowfield(self, flow_orig):
        if self.downsample_factor <= 1:
            return flow_orig
        
        downsampled_flow = F.avg_pool2d(
            flow_orig,
            kernel_size = self.downsample_factor,
            stride = self.downsample_factor
        )
        return downsampled_flow
    
    @torch.no_grad()
    def run(self):
        '''
        Runs optical flow and returns a dict of:

        'diff' : mean pixel-wise difference (motion energy) for each ROI (in px / pframe)
        'mag' : mean pixel magnitude
        'ang'
        '''
        frame_idx = self.start + 1  # we have frame0, frame1 prepared
        cur, prev = 1, 0       # buffer indices
        max_frames = max(0, self.nframes - (self.start + 1))
        pbar = tqdm(total=max_frames)

        n_masks = int(self.maks_gpu.shape[0])
        mags_t  = torch.empty((max_frames, n_masks), device=self.device, dtype=torch.float32)
        angs_t  = torch.empty((max_frames, n_masks), device=self.device, dtype=torch.float32)
        diffs_t = torch.empty((max_frames, n_masks), device=self.device, dtype=torch.float32)
        t = 0

        # Downsampling for saving the vectorfield
        H, W = self.height, self.width
        h_dwnsampld = H // self.downsample_factor
        w_dwnsampld = W // self.downsample_factor
        downsampled_flows_t  = torch.empty((max_frames, 2, h_dwnsampld, w_dwnsampld), device=self.device, dtype=torch.float32)

        while True:
            ret, fn = self.cap.read()
            frame_idx += 1
            if not ret or frame_idx >= self.nframes:
                break
            nxt = prev  # reuse the oldest slot
            host_nxt = self._to_pinned(fn)
            with torch.cuda.stream(self.copy_stream):
                self.dev_buffers[nxt].copy_(host_nxt, non_blocking=True)


            with torch.amp.autocast(self.device, dtype=torch.float16), torch.cuda.stream(self.comp_stream):
                prev_frame = self.dev_buffers[prev].float() / 255.0
                cur_frame = self.dev_buffers[cur].float() / 255.0

                H, W = prev_frame.shape[-2], prev_frame.shape[-1]
                prev_frame_padded, (ph, pw) = self._pad(prev_frame)
                cur_frame_padded,  _        = self._pad(cur_frame)

                flow = self.model(prev_frame_padded, cur_frame_padded, num_flow_updates=self.niters)
                flow = flow[0] if isinstance(flow, (list, tuple)) else flow
                if ph or pw:
                    flow = self._undo_pad(flow, H, W)
                downsampled_flow = self._downsample_flowfield(flow.float())
                flow_x = flow[:, 0]
                flow_y = flow[:, 1]
                mag = torch.sqrt(flow_x * flow_x + flow_y * flow_y)
                # ang = torch.atan2(flow_y, flow_x)
                ang = torch.remainder(torch.atan2(flow_y, flow_x), 2 * torch.pi)

                diff = (self._rgb_to_gray(cur_frame) - self._rgb_to_gray(prev_frame)).abs()

                mean_mag  = (mag * self.maks_gpu).sum(dim=(2,3))[:,0] / self.px_per_mask            # [K]
                mean_ang  = (ang * self.maks_gpu).sum(dim=(2,3))[:,0] / self.px_per_mask
                mean_diff = (diff * self.maks_gpu).sum(dim=(2,3))[:,0] / self.px_per_mask
            
            # GPU still busy, continue CPU image reading
            pbar.update(1)

            # Ensure compute finished for this iteration before collecting results
            self.comp_stream.synchronize()
            mags_t[t].copy_(mean_mag)
            angs_t[t].copy_(mean_ang)
            diffs_t[t].copy_(mean_diff)

            # downsampled flows have the shape [1, 2, w, h], we need [2, w, h]. That's what squeeze does here.
            downsampled_flows_t[t].copy_(downsampled_flow.squeeze(0))
            t+=1

        torch.cuda.synchronize()
        self.cap.release()
        out = dict(
            diff=diffs_t[:t].cpu().numpy().astype(np.float32),
            mag=mags_t[:t].cpu().numpy().astype(np.float32),
            ang=angs_t[:t].cpu().numpy().astype(np.float32),
            flow_grid=downsampled_flows_t[:t].float().cpu().numpy()
        )
        return out


class FarnebackOF(BaseOF):
    def __init__(self, config_args=None, n_iters=3):
        self.args = config_args or dict(
            numLevels=5, pyrScale=.5, fastPyramids=True, winSize=25,
            numIters=n_iters, polyN=5, polySigma=1.2, flags=0
        )
        self.of = cv2.cuda_FarnebackOpticalFlow.create(**self.args)
        self.compute_stream = cv2.cuda.Stream()
        self.copy_stream = cv2.cuda.Stream()
        self.masks_np = None
        self.maks_gpu = []
        self.px_per_mask = None
        self.shape = None

        self.nframes = 0
        self.fps = 0
        self.height = 0
        self.width = 0

        self.gpu_bgr = None
        self.gpu_gray = None
        self.gpu_gray_f32 = None

        self.save_vectors = True
        self.downsample_factor = 8

        self.ready_evt = cv2.cuda.Event()
        
        super().__init__()
    
    def set_masks(self, masks: list[np.ndarray]):
        '''
        Uploads fixed ROI masks (before applying them to the Optical Flow output) to the GPU
        '''
        self.masks_np = [m.astype('f4') for m in masks]
        self.px_per_mask = np.array([m.sum() for m in masks], dtype='f4')
        for m in self.masks_np:
            m_gpu = cv2.cuda_GpuMat()
            m_gpu.upload(m, stream=self.copy_stream)
            self.maks_gpu.append(m_gpu)
        self.copy_stream.waitForCompletion()
        print("Masks are uploaded to GPU")
    
    def video_info(self):
        '''
        Just returns basic video info (#frames, fps, height, width)
        '''
        return self.nframes, self.fps, self.height, self.width

    def open(self, video_path: str, start=0, end=None):
        '''
        Opens a video file and uploads the first two frames to the GPU
        Blocks until the upload is finished
        '''
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f'cannot open video {video_path}')
        self.start = start
        if self.start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.nframes = total if end is None else min(end, total)

        # get first 2 frames on cpu
        ret, frame0 = self.cap.read()
        if not ret:
            raise RuntimeError("couldn't read frame 0")
        ret, frame1 = self.cap.read()
        if not ret:
            self.cap.release()
            raise RuntimeError("couldn't read frame 0")
        self.shape = frame0.shape
        rows, cols = self.shape[:2]
        self.height, self.width = rows, cols
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.host_buf = np.empty((rows, cols, 3), dtype=np.uint8)
        cv2.cuda.registerPageLocked(self.host_buf)  # Holds the buffer in CPU RAM

        # prealocate GPU buffers (for efficiancy reasons)
        self.gpu_bgr = [cv2.cuda_GpuMat(rows, cols, cv2.CV_8UC3) for _ in range(2)]
        self.gpu_gray = [cv2.cuda_GpuMat(rows, cols, cv2.CV_8UC1) for _ in range(2)]
        self.gpu_gray_f32 = [cv2.cuda_GpuMat(rows, cols, cv2.CV_32FC1) for _ in range(2)]

        # upload the first 2 frames
        self.host_buf[:] = frame0
        self.gpu_bgr[0].upload(self.host_buf, stream=self.copy_stream)
        self.host_buf[:] = frame1
        self.gpu_bgr[1].upload(self.host_buf, stream=self.copy_stream)
        cv2.cuda.cvtColor(self.gpu_bgr[0], cv2.COLOR_BGR2GRAY, dst=self.gpu_gray[0], stream=self.copy_stream)
        cv2.cuda.cvtColor(self.gpu_bgr[1], cv2.COLOR_BGR2GRAY, dst=self.gpu_gray[1], stream=self.copy_stream)
        self.gpu_gray_f32[0] = self.gpu_gray[0].convertTo(cv2.CV_32F, stream=self.copy_stream)
        self.gpu_gray_f32[1] = self.gpu_gray[1].convertTo(cv2.CV_32F, stream=self.copy_stream)

        # frame 0 and frame 1 ready for computation
        self.ready_evt.record(self.copy_stream)
        self.compute_stream.waitEvent(self.ready_evt)

    def run(self):
        rows, cols = self.shape[:2]
        frame_idx = self.start + 1  # we have frame0, frame1 already prepared (in open())
        cur, prev = 1, 0       # buffer indices
        out_idx = 0

        nm = len(self.maks_gpu)
        max_frames = max(0, self.nframes - (self.start + 1))

        # GPU Buffers for optical flow
        flow = cv2.cuda_GpuMat(rows, cols, cv2.CV_32FC2)
        flow_x = cv2.cuda_GpuMat(rows, cols, cv2.CV_32FC1)
        flow_y = cv2.cuda_GpuMat(rows, cols, cv2.CV_32FC1)
        # GPU Buffers for the ROI-wise angle / magnitude / difference calculations
        sum_mag64  = [cv2.cuda_GpuMat(1, 1, cv2.CV_64F) for _ in range(nm)]
        sum_ang64  = [cv2.cuda_GpuMat(1, 1, cv2.CV_64F) for _ in range(nm)]
        sum_diff64 = [cv2.cuda_GpuMat(1, 1, cv2.CV_64F) for _ in range(nm)]
        sum_mag32  = [cv2.cuda_GpuMat(1, 1, cv2.CV_32F) for _ in range(nm)]
        sum_ang32  = [cv2.cuda_GpuMat(1, 1, cv2.CV_32F) for _ in range(nm)]
        sum_diff32 = [cv2.cuda_GpuMat(1, 1, cv2.CV_32F) for _ in range(nm)]

        # CPU Buffers for ROI-wise angles / magnitudes / differences
        mag_host  = np.empty((max_frames, nm), dtype=np.float32)
        ang_host  = np.empty((max_frames, nm), dtype=np.float32)
        diff_host = np.empty((max_frames, nm), dtype=np.float32)
        # Page lock just for performance (these buffers always stay in RAM)
        cv2.cuda.registerPageLocked(mag_host)
        cv2.cuda.registerPageLocked(ang_host)
        cv2.cuda.registerPageLocked(diff_host)

        d_h = rows // self.downsample_factor
        d_w = cols // self.downsample_factor
        
        # GPU Buffer for the current resized flow
        flow_small = cv2.cuda_GpuMat(d_h, d_w, cv2.CV_32FC2)
        # CPU Buffer (Pinned) for saving all frames
        flow_grid_host = np.empty((max_frames, d_h, d_w, 2), dtype=np.float32)
        cv2.cuda.registerPageLocked(flow_grid_host)

        pbar = tqdm(total=max_frames)

        while True:
            self.of.calc(self.gpu_gray[prev], self.gpu_gray[cur], flow, stream=self.compute_stream)
            cv2.cuda.resize(flow, (d_w, d_h), dst=flow_small, stream=self.compute_stream)
            flow_small.download(stream=self.copy_stream, dst=flow_grid_host[out_idx])
            cv2.cuda.split(flow, [flow_x, flow_y], stream=self.compute_stream)
            mag, ang = cv2.cuda.cartToPolar(flow_x, flow_y, angleInDegrees=False, stream=self.compute_stream)
            diff = cv2.cuda.absdiff(self.gpu_gray_f32[cur], self.gpu_gray_f32[prev], stream=self.compute_stream)

            # reads next frame asynchronously on CPU while GPU is busy
            ret, fn = self.cap.read()
            if not ret:
                break
            # upload and convert next frame on copy_stream
            nxt = prev
            self.host_buf[:] = fn  # memcpy into pinned host buffer
            self.gpu_bgr[nxt].upload(self.host_buf, stream=self.copy_stream)  # async GPU upload frame
            cv2.cuda.cvtColor(self.gpu_bgr[nxt], cv2.COLOR_BGR2GRAY,
                            dst=self.gpu_gray[nxt], stream=self.copy_stream)
            self.gpu_gray_f32[nxt] = self.gpu_gray[nxt].convertTo(cv2.CV_32F, stream=self.copy_stream)


            # while the frame is uploading, we continue computing on the GPU
            for i, (gm, _) in enumerate(zip(self.maks_gpu, self.px_per_mask)):
                mm = cv2.cuda.multiply(mag,  gm, stream=self.compute_stream)
                aa = cv2.cuda.multiply(ang,  gm, stream=self.compute_stream)
                dd = cv2.cuda.multiply(diff, gm, stream=self.compute_stream)

                cv2.cuda.calcAbsSum(mm, sum_mag64[i],  mask=None, stream=self.compute_stream)
                cv2.cuda.calcAbsSum(aa, sum_ang64[i],  mask=None, stream=self.compute_stream)
                cv2.cuda.calcAbsSum(dd, sum_diff64[i], mask=None, stream=self.compute_stream)

                # convert to CV_32F on GPU
                sum_mag32[i] = sum_mag64[i].convertTo(cv2.CV_32F, stream=self.compute_stream)
                sum_ang32[i] = sum_ang64[i].convertTo(cv2.CV_32F, stream=self.compute_stream)
                sum_diff32[i] = sum_diff64[i].convertTo(cv2.CV_32F, stream=self.compute_stream)

                # async GPU to CPU copy, calculations on GPU can already continue!
                mag_host[out_idx:out_idx+1, i] = sum_mag32[i].download(stream=self.copy_stream)
                ang_host[out_idx:out_idx+1, i] = sum_ang32[i].download(stream=self.copy_stream)
                diff_host[out_idx:out_idx+1, i] = sum_diff32[i].download(stream=self.copy_stream)
            
            pbar.update(1)
            frame_idx += 1
            out_idx += 1
            if frame_idx >= self.nframes:
                break

            # signal copy done, make compute stream wait before next iteration uses it
            self.ready_evt.record(self.copy_stream)
            self.compute_stream.waitEvent(self.ready_evt)

            # rotate buffers: (prev,cur) <= (cur,nxt)
            prev, cur = cur, nxt

        # ensure all GPU work finished before returning all results
        self.copy_stream.waitForCompletion()
        self.compute_stream.waitForCompletion()
        self.cap.release()
        pbar.close()
        px = np.maximum(np.asarray(self.px_per_mask, dtype=np.float32), 1.0)
        mags  = (mag_host[:out_idx, :]  / px[None, :]).astype(np.float32)
        angs  = (ang_host[:out_idx, :]  / px[None, :]).astype(np.float32)
        diffs = (diff_host[:out_idx, :] / px[None, :]).astype(np.float32)
        flow_grid = flow_grid_host[:out_idx].transpose(0, 3, 1, 2)
        return dict(
            diff=np.array(diffs, dtype=np.float32),
            mag =np.array(mags,  dtype=np.float32),
            ang =np.array(angs,  dtype=np.float32),
            flow_grid=flow_grid,
        )
    

def load_masks(path):
    data = np.load(path, allow_pickle=True)
    return data['masks']

def facemotion(videopath, masks, backend: BaseOF, start=0, end=None):
    backend.set_masks(masks)
    backend.open(videopath, start=start, end=end)
    start = time.time()
    out = backend.run()  # returns dict of CPU arrays
    end = time.time()
    nframes, fps, height, width = backend.video_info()
    print(f"processing took {end - start} seconds, frames_processed: {nframes}, normalized: {nframes / (end - start)}, resolution: {width} x {height}, fps: {fps}")
    motion = np.hstack([out['diff'], out['mag'], out['ang']])
    cols = ['MotionEnergy_Nose','MotionEnergy_Whiskerpad','MotionEnergy_Mouth','MotionEnergy_Cheek',
            'OFmag_Nose','OFmag_Whiskerpad','OFmag_Mouth','OFmag_Cheek',
            'OFang_Nose','OFang_Whiskerpad','OFang_Mouth','OFang_Cheek']
    return pd.DataFrame(motion, columns=cols)

