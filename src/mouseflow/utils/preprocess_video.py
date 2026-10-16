import cv2
from tqdm import tqdm
from pathlib import Path

def flip_vid(vid_orig, horizontal=False, vertical=False):
    cap = cv2.VideoCapture(vid_orig)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vidlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_flipped_name = (vid_orig[:-4]+'_flipped.mp4')
    if Path(vid_flipped_name).exists():
        print("video already flipped, skipping...")
        return vid_flipped_name
    vid_flipped = cv2.VideoWriter(vid_flipped_name, cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (frame_width, frame_height))
    
    print(f'Flipping video {vid_orig}')
    with tqdm(total=vidlength) as pbar:
        while cap.isOpened():
            ret, current_frame = cap.read()
            if current_frame is None:
                break
            if horizontal:
                current_frame = cv2.flip(current_frame, 1)
            if vertical:
                current_frame = cv2.flip(current_frame, 0)
            vid_flipped.write(current_frame)
            pbar.update(1)

    return vid_flipped_name

def crop_vid(vid_orig, crop):
    crop = [int(c) for c in crop]  # ensuring only integers in list
    cap = cv2.VideoCapture(vid_orig)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    vidlength = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_cropped = cv2.VideoWriter(vid_orig[:-4]+'_cropped.mp4', cv2.VideoWriter_fourcc(*"mp4v"),
                                  fps, (frame_width, frame_height))
    
    print('Cropping video...')
    with tqdm(total=vidlength) as pbar:
        while cap.isOpened():
            _, current_frame = cap.read()
            if current_frame is None:
                break
            cropped_frame = current_frame[crop[0]:crop[1], crop[2]:crop[3]]
            vid_cropped.write(cropped_frame)
            pbar.update(1)

    return vid_orig[:-4]+'_cropped.mp4'

def get_video_for_marker(marker_file: Path, dlc_dir: Path):
        VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v")
        """
        Return the single video in dlc_dir matching the marker file.
        Assumes 0 or 1 matching video per marker.
        Match rule: video filename starts with the marker name portion before 'DLC' (case-insensitive).
        """

        marker_file = Path(marker_file)
        dlc_dir = Path(dlc_dir)

        if marker_file.suffix.lower() != ".h5":
            raise ValueError(f"Expected .h5 marker file, got: {marker_file}")

        base = marker_file.stem
        i = base.lower().find("dlc")
        prefix = base[:i] if i != -1 else base
        prefix = prefix.rstrip("_-. ")

        candidates = [
            p for p in dlc_dir.parent.iterdir()
            if (p.is_file()
                and p.suffix.lower() in VIDEO_EXTS
                and p.name.startswith(prefix)
                and "labeled" not in p.name
            )
        ]

        if not candidates:
            raise FileNotFoundError(f"No video starting with '{prefix}' in {dlc_dir.parent}")

        if len(candidates) > 1:
            raise RuntimeError(
                f"Expected at most one video for '{marker_file.name}', found: {[c.name for c in candidates]}")

        return candidates[0]


def video_info(video_path: Path | str):
        """
        Open a video and return (fps, width, height, cap).
        Caller owns 'cap' and should release it when done.
        Raises OSError if the file cannot be opened or has invalid properties.
        """
        video_path = str(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise OSError(f"Failed to open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if fps <= 0:
            raise OSError(f"Invalid FPS ({fps}) for: {video_path}")

        if width <= 0 or height <= 0:
            cap.release()
            raise OSError(f"Invalid frame size {width}x{height} for: {video_path}")

        return float(fps), width, height, cap