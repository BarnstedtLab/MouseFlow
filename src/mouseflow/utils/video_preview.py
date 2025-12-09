'''
Author: @Jakob Faust
Date: 01.12.2025
'''


import abc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm


@dataclass
class FrameContext:
    """Holds the state for the current frame being processed (which is the same for all video panels)."""
    frame_idx: int
    frame_bgr_face: np.ndarray
    frame_gray_face: np.ndarray
    frame_bgr_body: np.ndarray
    frame_gray_body: np.ndarray
    data_row: pd.Series
    face_masks: Optional[np.ndarray] = None

class BasePanel(abc.ABC):
    """Abstract base class for any visualization panel."""
    def __init__(self, key: str, title: str = ""):
        self.key = key
        self.ax = None
        self.title = title

    def setup(self, ax_dict: Dict[str, plt.Axes]):
        """Called once at start to link the axes."""
        self.ax = ax_dict[self.key]
        if self.title:
            self.ax.set_title(self.title)
        self.ax.axis('off')

    @abc.abstractmethod
    def update(self, ctx: FrameContext):
        """Called every frame to draw content."""
        pass

### Implementation for the actual video panels satrt here

# ------------------ Body keypoints panel -----------------------------------
class KeypointPanel(BasePanel):
    """Displays the main video with DLC overlays and skeleton."""
    def __init__(self, key, dataframe: pd.DataFrame, conf_thresh=0.99, plot_type='face'):
        super().__init__(key)
        self.dataframe = dataframe
        self.scat_artist = None
        self.conf_thresh = conf_thresh
        self.plot_type = plot_type
    
    def setup(self, ax_dict):
        super().setup(ax_dict)
        self.im_artist = self.ax.imshow(np.zeros((1,1,3)), animated=True)
        self.ax.axis('off')

    def update(self, ctx: FrameContext):
        if self.plot_type == 'face':
            rgb = cv2.cvtColor(ctx.frame_bgr_face, cv2.COLOR_BGR2RGB)
        elif self.plot_type == 'body':
            rgb = cv2.cvtColor(ctx.frame_bgr_body, cv2.COLOR_BGR2RGB)
        self.im_artist.set_data(rgb)
        
        if ctx.frame_idx == 0: 
            h, w = rgb.shape[:2]
            self.im_artist.set_extent((0, w, h, 0))
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
        row = self.dataframe.iloc[ctx.frame_idx]

        if self.scat_artist:
            self.scat_artist.remove()
            self.scat_artist = None
        
        vals = row.values
        x_pts, y_pts = [], []
        
        h, w = rgb.shape[:2]

        for i in range(0, len(vals) - 2, 3):
            x, y, lik = vals[i], vals[i+1], vals[i+2]
            if lik > self.conf_thresh and 0 <= x <= w and 0 <= y <= h:
                x_pts.append(x)
                y_pts.append(y)

        if x_pts:
            self.scat_artist = self.ax.scatter(x_pts, y_pts, s=20, c='cyan', alpha=0.6)

# ------------------ Optical flow panel -----------------------------------

class OpticalFlowPanel(BasePanel):
    """
    Displays the video frame with vector arrows overlayed.
    Arrows are colored by their angle (direction).
    """
    def __init__(self, key, flow, grid_step=8, arrow_scale=0.6):
        super().__init__(key)
        self.flow = flow       # Shape: [Time, 2, H, W]
        self.grid_step = grid_step
        self.arrow_scale = arrow_scale
        self.t = 0
        
        self.im_artist = None
        self.quiver_artist = None
        self.is_first_frame = True

    def setup(self, ax_dict):
        super().setup(ax_dict)
        self.im_artist = self.ax.imshow(np.zeros((1,1,3)), cmap='gray', vmin=0, vmax=255)
        self.ax.axis('off')

    def update(self, ctx: FrameContext):
        bg_img = ctx.frame_gray_face // 2
        self.im_artist.set_data(bg_img)
        if self.is_first_frame:
            h, w = bg_img.shape
            self.im_artist.set_extent((0, w, h, 0))
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
            
            # Positions, where we actually wanna plot the arrows
            flow = self.flow[ctx.frame_idx, :, :, :]
            grid_h, grid_w = flow.shape[1], flow.shape[2]
            
            x_coords = np.arange(0, grid_w) * self.grid_step + (self.grid_step / 2)
            y_coords = np.arange(0, grid_h) * self.grid_step + (self.grid_step / 2)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            self.quiver_artist = self.ax.quiver(
                X, Y, 
                np.zeros_like(X), np.zeros_like(Y), # Initial U, V (zeros)
                np.zeros_like(X),                   # Initial Colors (zeros)
                cmap='hsv',                         # Circular colormap for angles
                pivot='mid', 
                units='xy', 
                scale=self.arrow_scale,             # Controls length
                width=1.,
                headwidth=2
            )
            self.is_first_frame = False

        if ctx.frame_idx < self.flow.shape[0]:
            # Get flow for this frame (shape [2, H, W])
            flow_slice = self.flow[ctx.frame_idx, :, :, :]
            u = flow_slice[0]
            v = flow_slice[1]

            angles = np.arctan2(v, u)
            
            # Update Data
            self.quiver_artist.set_UVC(u, v, angles)


# ------------------ Pupil panel -----------------------------------

class PupilPanel(BasePanel):
    """Crops the eye region and shows pupil fit."""
    def __init__(self, key, width=240, height=180):
        super().__init__(key)
        self.w = width
        self.h = height
    
    def setup(self, ax_dict):
        super().setup(ax_dict)
        # Placeholder image, will be replaced in the update() function
        self.im_artist = self.ax.imshow(np.zeros((self.h, self.w)), cmap='gray', vmin=0, vmax=255)
        # Placeholder circle, will be replaced in the update() function
        self.circ_artist = plt.Circle((-100, -100), 10, color='red', fill=False, lw=1)
        self.ax.add_patch(self.circ_artist)
        self.ax.axis('off')

    def update(self, ctx: FrameContext):
        try:
            # (px, py) =  midpoint of pupil
            px = int(ctx.data_row[('smooth', 'PupilX')])
            py = int(ctx.data_row[('smooth', 'PupilY')])
            pr = ctx.data_row[('smooth', 'PupilDiam')]
        except (KeyError, ValueError):
            return

        # crop image around the eye
        x0 = max(0, px - self.w // 2) # width // 2 to the left of the pupil center and to the right 
        y0 = max(0, py - self.h // 2) # height // 2 above the pupil center and below
        
        crop = ctx.frame_gray_face[y0:y0+self.h, x0:x0+self.w]
        if crop.shape[0] != self.h or crop.shape[1] != self.w: # frame size somehow does not match, better ignore frame
             pass 
        else:
            # We apply the data to the current frame
            self.im_artist.set_data(crop)

        # Fast pupil circle update (we do not draw a new circle, buit just shift and rescale the old one)
        self.circ_artist.center = (px - x0, py - y0)
        self.circ_artist.radius = pr


class TracePanel(BasePanel):
    """
    Panel that shows the raw data traces (No normalization, no stacking)
    """
    def __init__(self, key, df_full: pd.DataFrame, cols_to_plot: List[str], window_sec=10, fps=100):
        super().__init__(key)
        self.df = df_full
        self.cols = cols_to_plot
        self.window_frames = int(window_sec * fps)
        self.half_window = self.window_frames // 2
        self.fps = fps
        
        # Cache for artists
        self.lines = []
        self.vline = None


        all_values = self.df[self.cols].values
        
        y_min_raw = np.nanmin(all_values)
        y_max_raw = np.nanmax(all_values)
        
        pad = (y_max_raw - y_min_raw) * 0.05 if y_max_raw != y_min_raw else 1.0
        
        self.y_min = y_min_raw - pad
        self.y_max = y_max_raw + pad

    def setup(self, ax_dict):
        super().setup(ax_dict)
        self.ax.axis('on')
        self.ax.set_xlabel("Time (s)")
        
        self.ax.set_ylim(self.y_min, self.y_max)
        
        for col in self.cols:
            line, = self.ax.plot([], [], label=str(col)) 
            self.lines.append(line)
        
        # 0 line
        self.vline = self.ax.axvline(0, color='white', linestyle=':')

        self.ax.legend(loc='upper right', fontsize='small')

    def update(self, ctx: FrameContext):
        # Calculate indices
        start = max(0, ctx.frame_idx - self.half_window)
        end = min(len(self.df), ctx.frame_idx + self.half_window)
        
        x_data = np.arange(start, end)
        
        for i, col in enumerate(self.cols):
            y_data = self.df[col].iloc[start:end].values
            self.lines[i].set_data(x_data, y_data)
            
        self.vline.set_xdata([ctx.frame_idx, ctx.frame_idx])
        
        self.ax.set_xlim(ctx.frame_idx - self.half_window, ctx.frame_idx + self.half_window)
        
        ticks = np.linspace(ctx.frame_idx - self.half_window, ctx.frame_idx + self.half_window, 5)
        self.ax.set_xticks(ticks)
        self.ax.set_xticklabels([f"{(t - ctx.frame_idx)/self.fps:.1f}s" for t in ticks])



class Coordinator:
    def __init__(self, video_path_face: str, video_path_body: str, output_path: str, layout: List[List[str]], panels: List[BasePanel]):
        self.video_path_face = video_path_face
        self.video_path_body = video_path_body
        self.output_path = output_path
        self.layout = layout
        self.panels = panels
        self.cap_face = cv2.VideoCapture(video_path_face)
        self.cap_body = cv2.VideoCapture(video_path_body)
        self.max_frames = int(self.cap_face.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap_face.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap_face.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap_face.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    def run(self, start_frame=0, duration_frames=None, df_data=None):
        plt.style.use('dark_background')
        fig, axd = plt.subplot_mosaic(self.layout, figsize=(16, 9), constrained_layout=True)
        
        # Prepare the individual panels 
        for panel in self.panels:
            panel.setup(axd)

        if duration_frames is None:
            duration_frames = int(max(0, self.max_frames - start_frame))

        # Prepare the video writing
        fig.canvas.draw()
        canvas_w, canvas_h = fig.canvas.get_width_height()
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (canvas_w, canvas_h))
        
        # Processing Loop
        self.cap_face.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self.cap_body.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        print(f"Processing {duration_frames} frames to {self.output_path}...")
        
        for i in tqdm(range(duration_frames)):
            ok_face, frame_face = self.cap_face.read()
            ok_body, frame_body = self.cap_body.read()
            if not ok_face or not ok_body: break
            
            abs_idx = start_frame + i
            
            ctx = FrameContext(
                frame_idx=abs_idx,
                frame_bgr_face=frame_face,
                frame_gray_face=cv2.cvtColor(frame_face, cv2.COLOR_BGR2GRAY),
                frame_bgr_body=frame_body,
                frame_gray_body=cv2.cvtColor(frame_body, cv2.COLOR_BGR2GRAY),
                data_row=df_data.iloc[abs_idx] if df_data is not None else None
            )
            
            # Update all panels
            for panel in self.panels:
                panel.update(ctx)
                
            # Render to Video
            fig.canvas.draw()
            
            # Convert canvas to an image for OpenCV
            img_plot = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_plot = img_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_plot = cv2.cvtColor(img_plot, cv2.COLOR_RGBA2BGR)
            
            writer.write(img_plot)
    
        writer.release()
        self.cap_face.release()
        self.cap_body.release()
        plt.close(fig)
        print("Done.")


def find_file(analysis_dir, pattern, exclude_pattern=None):
    analysis_path = Path(analysis_dir)

    files = list(analysis_path.glob(pattern))

    if exclude_pattern:
        files = [f for f in files if exclude_pattern not in f.name]

    if not files:
        return None
    
    files.sort()
    return str(files[0])


def fetch_files(face_video_file, body_video_file, analysis_dir):
    body_video_path = Path(body_video_file)
    face_video_path = Path(face_video_file)

    body_video_name = body_video_path.stem
    face_video_name = face_video_path.stem

    dlc_body_file = find_file(analysis_dir, f"{body_video_name}*DLC*.h5", exclude_pattern="mouseflow")
    dlc_face_file = find_file(analysis_dir, f"{face_video_name}*DLC*.h5", exclude_pattern="mouseflow")

    mf_file_face = find_file(analysis_dir, f"{face_video_name}*mouseflow.h5")
    of_file_face = find_file(analysis_dir, f"{face_video_name}*optical_flow_grid.npz")

    return dlc_body_file, dlc_face_file, mf_file_face, of_file_face



def create_video_preview(mf_file_face=None, keypoint_file_face=None, keypoint_file_body=None, video_file_face=None, video_file_body=None, optical_flow_file=None, output_file=None, start_frame=0, n_frames=None):
    
    panels_list = []

    if optical_flow_file:
        of_grid = np.load(optical_flow_file)["flow"]
        opticalflow_panel = OpticalFlowPanel(key='optflow', flow=of_grid, arrow_scale=2.)
        panels_list.append(opticalflow_panel)
    if keypoint_file_face:
        keypoint_face_df = pd.read_hdf(keypoint_file_face)
        keypoint_panel_face = KeypointPanel(key='face_keypoints', dataframe=keypoint_face_df, conf_thresh=0.25, plot_type='face')
        pupil_panel = PupilPanel(key='pupil')
        panels_list.append(keypoint_panel_face)
        panels_list.append(pupil_panel)
    if keypoint_file_body:
        keypoint_body_df = pd.read_hdf(keypoint_file_body)
        keypoint_panel_body = KeypointPanel(key='body_keypoints', dataframe=keypoint_body_df, conf_thresh=0.25, plot_type='body')
        panels_list.append(keypoint_panel_body)
    if mf_file_face:
        mf_face_df = pd.read_hdf(mf_file_face, key='face')
        trace_panel = TracePanel(
            key='traces', 
            df_full=mf_face_df, 
            cols_to_plot=[
                ('smooth', 'MotionEnergy_Mouth'),
                ('smooth', 'OFang_Whiskerpad')
            ]
        )
        panels_list.append(trace_panel)

    # layout_def = [
    #     [keypoint_panel_face.key, keypoint_panel_body.key],
    #     [opticalflow_panel.key, pupil_panel.key],
    #     [trace_panel.key, trace_panel.key]
    # ]


    main_keys = []
    trace_key = None

    for panel in panels_list:
        if isinstance(panel, TracePanel): 
            trace_key = panel.key
        else:
            main_keys.append(panel.key)

    # Build the Main Grid (2 columns wide)
    layout_def = []
    cols = 2

    for i in range(0, len(main_keys), cols):
        row = main_keys[i:i + cols]        
        if len(row) == cols:
            layout_def.append(row)
        else:
            layout_def.append([row[0]] * cols)

    if trace_key:
        layout_def.append([trace_key] * cols)

    if not layout_def:
        print("No panels to plot.")
    else:
        print("Generated Layout:", layout_def)
    
    panels_list = [
        keypoint_panel_face, keypoint_panel_body, opticalflow_panel, pupil_panel, trace_panel
    ]


    viz = Coordinator(
        video_path_face=video_file_face,
        video_path_body=video_file_body,
        output_path=output_file if output_file is not None else Path(video_file_face).stem + "_preview.mp4",
        layout=layout_def,
        panels=panels_list
    )
    
    viz.run(start_frame=start_frame, duration_frames=n_frames, df_data=mf_face_df)

def generate_preview(vid_file_face, vid_file_body, analysis_dir=None, out_file=None):
    if analysis_dir is None:
        analysis_dir = str(Path(vid_file_face).parent / "mouseflow")
    
    video_file_face_path = Path(vid_file_face)
    if out_file is None:
        out_file = str(video_file_face_path.parent / f"{video_file_face_path.stem}_preview.mp4")

    dlc_file_body, dlc_file_face, mf_file_face, grid_file = fetch_files(
        vid_file_face,
        vid_file_body,
        analysis_dir)
    
    create_video_preview(
        mf_file_face=mf_file_face, 
        keypoint_file_face=dlc_file_face,
        keypoint_file_body=dlc_file_body,
        video_file_face=vid_file_face,
        video_file_body=vid_file_body,
        optical_flow_file=grid_file,
        output_file=out_file,
        n_frames=100)
    