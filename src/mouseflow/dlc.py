import os
import glob
from mouseflow.utils.preprocess_video import flip_vid, crop_vid
from mouseflow.utils.pytorch_utils import config_pytorch
from mouseflow.apply_models import LPDetector, DLCDetector


def runDLC(
    vid_dir=os.getcwd(),
    face_cfg=[], face_weights=None,
    body_cfg=None, body_weights=None,
    face_model='DLC', body_model='DLC',     # 'DLC' for *.pt, 'LP' for *.ckpt
    facekey='face', bodykey='body',
    batch='all', overwrite=False, filetype='.avi',
    vid_output=True, body_facing='right', face_facing='left',
    face_crop=None, body_crop=None
):
    # vid_dir defines directory to detect face/body videos, standard: current working directory
    # facekey defines unique string that is contained in all face videos. If None, no face videos will be considered.
    # bodykey defines unique string that is contained in all body videos. If None, no body videos will be considered.
    # dgp defines whether to use DeepGraphPose (if True), otherwise resorts to DLC
    # batch defines how many videos to analyse ('all' for all, integer for the first n videos)
    # face/body_crop allows initial cropping of video in the form [x_start, x_end, y_start, y_end]

    #  To evade cuDNN error message:
    device = config_pytorch(benchmark=True, deterministic=False)

    # check where marker models are located, download if not present
    # dlc_faceyaml, dlc_bodyyaml = apply_models.download_models(
    #     models_dir, facemodel_name, bodymodel_name)
    # face_engine = apply_models.detect_engine(dlc_faceyaml)
    # body_engine = apply_models.detect_engine(dlc_bodyyaml)

    # identify video files
    facefiles = []
    bodyfiles = []
    if os.path.isfile(vid_dir):
        if facekey in vid_dir:
            facefiles = [vid_dir]
        elif bodykey in vid_dir:
            bodyfiles = [vid_dir]
        else:
            print(
                f'Need to pass <facekey> or <bodykey> argument to classify video {vid_dir}.')
    if facekey == True:
        facefiles = [vid_dir]
    elif bodykey == True:
        bodyfiles = [vid_dir]
    elif facekey == '' or facekey == False or facekey == None:
        bodyfiles = glob.glob(os.path.join(vid_dir, '*'+bodykey+'*'+filetype))
    elif bodykey == '' or bodykey == False or bodykey == None:
        facefiles = glob.glob(os.path.join(vid_dir, '*'+facekey+'*'+filetype))
    else:
        facefiles = glob.glob(os.path.join(vid_dir, '*'+facekey+'*'+filetype))
        bodyfiles = glob.glob(os.path.join(vid_dir, '*'+bodykey+'*'+filetype))

    # cropping videos
    facefiles = [f for f in facefiles if '_cropped.*' not in f]  # sort out already cropped videos
    bodyfiles = [b for b in bodyfiles if '_cropped.*' not in b]  # sort out already cropped videos
    if face_crop:
        facefiles_cropped = []
        for vid in facefiles:
            facefiles_cropped.append(crop_vid(vid, face_crop))
        facefiles = facefiles_cropped
    if body_crop:
        bodyfiles_cropped = []
        for vid in bodyfiles:
            bodyfiles_cropped.append(crop_vid(vid, body_crop))
        bodyfiles = bodyfiles_cropped

    # flipping videos
    facefiles = [f for f in facefiles if '_flipped.*' not in f]  # sort out already flipped videos
    bodyfiles = [b for b in bodyfiles if '_flipped.*' not in b]  # sort out already flipped videos
    if face_facing != 'left':
        facefiles_flipped = []
        for vid in facefiles:
            facefiles_flipped.append(flip_vid(vid, horizontal=True))
        facefiles = facefiles_flipped
    if body_facing != 'right':
        bodyfiles_flipped = []
        for vid in bodyfiles:
            bodyfiles_flipped.append(flip_vid(vid, horizontal=True))
        bodyfiles = bodyfiles_flipped

    # batch mode (if user specifies a number n, it will only process the first n files)
    try:
        batch = int(batch)
        facefiles = facefiles[:batch]
        bodyfiles = bodyfiles[:batch]
        print(f'Only processing first {batch} face and body videos...')
    except ValueError:
        pass

    # set directories
    if os.path.isdir(vid_dir):
        dir_out = os.path.join(vid_dir, 'mouseflow')
    else:
        dir_out = os.path.join(os.path.dirname(vid_dir), 'mouseflow')
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    # FACE
    if face_cfg and face_weights:
        print('facecfg and faceweights ok')
        for facefile in facefiles:
            print(f"file found: {facefile}")
            out_exists = glob.glob(os.path.join(dir_out, os.path.basename(facefile)[:-4] + '*.h5'))
            if out_exists and not overwrite:
                print(f'Skipping {os.path.basename(facefile)} (already labelled).')
                continue
            print("Applying FACE model:", face_cfg, "weights:", face_weights)
            if face_model == 'LP':
                det = LPDetector(face_cfg, face_weights)
            else:  # 'DLC'
                det = DLCDetector(face_cfg, face_weights, shuffle=1)
            det.detect_keypoints(facefile, dir_out, vid_output, overwrite)

    # BODY
    if body_cfg and body_weights:
        print('body and body ok')
        for bodyfile in bodyfiles:
            print(f"file found: {bodyfile}")
            out_exists = glob.glob(os.path.join(dir_out, os.path.basename(bodyfile)[:-4] + '*.h5'))
            if out_exists and not overwrite:
                print(f'Skipping {os.path.basename(bodyfile)} (already labelled).')
                continue
            print("Applying BODY model:", body_cfg, "weights:", body_weights)
            if body_model == 'LP':
                det = LPDetector(body_cfg, body_weights)
            else:
                det = DLCDetector(body_cfg, body_weights, shuffle=3)
            det.detect_keypoints(bodyfile, dir_out, vid_output, overwrite)

    # Apply DLC/DGP Model to each face video
    # for facefile in facefiles:
    #     print(f">>> PROCESSING FACE (engine: {face_engine})  file: {os.path.basename(facefile)}")
    #     if glob.glob(os.path.join(dir_out, os.path.basename(facefile)[:-4]+'*.h5')) and not overwrite:
    #         print(
    #             f'Video {os.path.basename(facefile)} already labelled. Skipping ahead...')
    #     else:
    #         print("Applying ", dlc_faceyaml, " on FACE video: ", facefile)
    #         # if face_engine == 'pytorch':        # DLC3
    #         #     apply_models.apply_dlc_pt(      
    #         #         filetype, vid_output, dlc_faceyaml, dir_out,
    #         #         facefile, overwrite, device=device)
            
    #         if face_model == 'LP':        # DLC3
    #             lpdetector = LPDetector(dlc_faceyaml, dlc_faceyaml) #TODO!!!
    #             lpdetector.detect_keypoints(facefile, dir_out, vid_output, overwrite)
    #         elif face_model == 'DLC':        # DLC3
    #             dlcdetector = DLCDetector(dlc_faceyaml, dlc_faceyaml) #TODO!!!
    #             dlcdetector.detect_keypoints(facefile, dir_out, vid_output, overwrite)
    #         else:
    #             raise RuntimeError("Make sure pytorch is installed on your system to run DLC.")

    # # Apply DLC/DGP Model to each body video
    # for bodyfile in bodyfiles:
    #     print(f">>> PROCESSING BODY (engine: {body_engine})  file: {os.path.basename(bodyfile)}")
    #     if glob.glob(os.path.join(dir_out, os.path.basename(bodyfile)[:-4]+'*.h5')) and not overwrite:
    #         print(
    #             f'Video {os.path.basename(bodyfile)} already labelled. Skipping ahead...')
    #     else:
    #         print("Applying ", dlc_bodyyaml, " on BODY video: ", bodyfile)
    #         # if body_engine == 'pytorch':        # DLC3
    #         #     apply_models.apply_dlc_pt(      
    #         #         filetype, vid_output, dlc_bodyyaml, dir_out,
    #         #         bodyfile, overwrite, device=device)
    #         if body_model == 'LP':        # DLC3
    #             lpdetector = LPDetector(dlc_bodyyaml, dlc_bodyyaml) #TODO!!!
    #             lpdetector.detect_keypoints(bodyfile, dir_out, vid_output, overwrite)
    #         elif body_model == 'DLC':        # DLC3
    #             dlcdetector = DLCDetector(dlc_bodyyaml, dlc_bodyyaml) #TODO!!!
    #             dlcdetector.detect_keypoints(bodyfile, dir_out, vid_output, overwrite)
    #         else:
    #             raise RuntimeError("Make sure pytorch is installed on your system to run DLC.")