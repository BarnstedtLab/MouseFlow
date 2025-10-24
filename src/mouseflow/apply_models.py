import glob
import os
import numpy as np
import gdown
import yaml


def detect_engine(cfg_path):
    """
    Return 'pytorch' if this config belongs to DLC 3,
    otherwise None.
    """
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    if (os.path.basename(cfg_path).startswith('pytorch_')   # DLC 3 puts a prefix
        or cfg.get('engine') == 'pytorch'):                 # or DLC 3 may embed key
        return 'pytorch'
    return None

def download_models(models_dir, facemodel_name, bodymodel_name):   
    if not os.path.exists(os.path.join(models_dir, facemodel_name)):
        dlc_face_url = 'https://drive.google.com/drive/folders/1_XPPyzaxMjQ901vJCwtv1g_h5DYWHM8j?usp=sharing'
        gdown.download_folder(dlc_face_url, models_dir, quiet=True, use_cookies=False)

    if not os.path.exists(os.path.join(models_dir, bodymodel_name)):
        dlc_body_url = 'https://drive.google.com/drive/folders/1_XPPyzaxMjQ901vJCwtv1g_h5DYWHM8j?usp=sharing'
        gdown.download_folder(dlc_body_url, models_dir, quiet=True, use_cookies=False)
    
    # PyTorch config (DLC 3)
    dlc_faceyaml = os.path.join(models_dir, facemodel_name, 'pytorch_config.yaml')
    dlc_bodyyaml = os.path.join(models_dir, bodymodel_name, 'pytorch_config.yaml')

    print(f"[download_models] → using facecfg = {dlc_faceyaml}")
    print(f"[download_models] → using bodycfg = {dlc_bodyyaml}")
    return dlc_faceyaml, dlc_bodyyaml


# TODO: change output file names
def apply_dgp(dlc_yaml, dir_out, vid_file, vid_output):
    from deepgraphpose.models.eval import estimate_pose, plot_dgp
    from deepgraphpose.models.fitdgp_util import get_snapshot_path
    snapshot_path, _ = get_snapshot_path('snapshot-0-step2-final--0', os.path.dirname(dlc_yaml), shuffle=1)
    if vid_output > 1:
        plot_dgp(vid_file,
                dir_out,
                proj_cfg_file=dlc_yaml,
                save_str=os.path.basename(dlc_yaml),
                dgp_model_file=str(snapshot_path),
                shuffle=1,
                dotsize=8,
                mask_threshold=0)
        print("DGP labels and labeled video saved in ", dir_out)
    else:
        estimate_pose(proj_cfg_file=dlc_yaml,
                            dgp_model_file=str(snapshot_path),
                            video_file=vid_file,
                            output_dir=dir_out,
                            shuffle=1,
                            save_pose=True,
                            save_str=os.path.basename(dlc_yaml),
                            new_size=None)
        print("DGP labels saved in ", dir_out)

# PyTorch DLC 3 model
def apply_dlc_pt(filetype, vid_output, dlc_yaml, dir_out, vid_file, overwrite, device=None):
    """
    Run DLC 3 (PyTorch) inference and optionally create a labeled video.
    Mirrors the TensorFlow helper but uses Engine.PYTORCH and torch.cuda.amp.
    """
    import torch
    from torch.cuda.amp import autocast
    import deeplabcut
    from deeplabcut.compat import analyze_videos
    from deeplabcut.core.engine import Engine
    from deeplabcut.utils.make_labeled_video import create_labeled_video as utils_create_labeled_video

    pt_cfg     = dlc_yaml                          # pytorch_config.yaml
    project_cfg = os.path.join(os.path.dirname(pt_cfg), "config.yaml")

    with torch.no_grad():
        with autocast():                           # mixed-precision for speed
            analyze_videos(
                config=project_cfg,
                videos=[vid_file],
                shuffle=3,
                videotype=filetype,
                destfolder=dir_out,
                engine=Engine.PYTORCH,             # backend flag
                modelprefix=os.path.dirname(pt_cfg),
                device=str(device)
            )
    print("DLC 3 (PyTorch) labels saved in", dir_out)

    if vid_output:
        utils_create_labeled_video(
            config=project_cfg,
            videos=[vid_file],
            videotype=filetype,
            shuffle=3,
            destfolder=dir_out,
            Frames2plot=None,       # could use np.arange(vid_output)
            draw_skeleton=False,
        )
        print("DLC 3 labeled video saved in", dir_out)