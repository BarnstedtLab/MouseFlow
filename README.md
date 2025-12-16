# MouseFlow

## Description
A Python toolbox to quantify facial and bodily movement in headfixed mice.
![Sample image](img/103_behaviour.gif)

## Installation
<img align="right" width="300" height="300" src="https://github.com/user-attachments/assets/7a01dfbf-11f8-422a-bbee-c270868c1361"/>
If you already have a working environment (with pytorch and deeplabcut installed) and do not want to create a new one, you can skip directly to step 6.
Otherwise, just follow along.
 
0. Make sure you have conda available on your system (otherwise, please follow the steps from the [official documentation](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html)
1. Download either the [linux setup file](enivronment_setup_linux.yaml) or the [windows setup file](enivronment_setup_windows.yaml), depending on your system.
3. Setup a fresh conda environment with all dependencies needed for mouseflow by running
```
conda env create -f path/to/downloaded/setup_file.yaml
```
4. Afterwards, you should have a new conda environment called `mouseflow_env`.
Activate it by running
```
conda activate mouseflow_env
```
5. Install deeplabcut (We need the `--pre` since we use Deeblabcut with Pytorch backend!)
```
pip install --pre deeplabcut
```
6. Finally, install the mouseflow package
```
pip install git+https://github.com/BarnstedtLab/MouseFlow --no-deps
```
(Users who directly came here without following the previous steps, please run the above command without `--no-deps`)

## Workflow
Currently, Mouseflow supports 3 main functions:

1. `detect_keypoints(vid_dir, face_key='face', body_key='body, face_model='DLC', body_model='DLC', filetype='.mp4')` finds all face and body videos in a directory of choice (vid_dir) and applies pre-trained DeepLabCut or LightningPose models (automatically downloaded during first use) on these videos.
Make sure all your body videos contain the provided `body_key` somewhere in their filename and the face videos the provided `face_key`.
If no videos are found, double check that your provided `filetype` matches your actual video files!
Note that LightingPose (`body_key='LP'`) will only work on linux machines.
3. `runMF(dlc_dir, save_optical_flow_vectors=True)` performs the analysis that lays the foundation for behavioral quantification.. Details are discussed [below](#Kinematics-and-optical-flow-extraction)
`dlc_dir` should point to the directory in which the keypoint analysis files from step 1 are saved.
if `save_optical_flow_vectors` is enabled, the optical flow output will be saved as a vectorfield (might generate large files if videos are long!) in a seperate .npz file. 
4. `generate_preview(mouseflow_file, face_keypoint_file, face_video_file, optical_flow_file)` generates a preview video (similar to the one shown above) allowing for a quick first visual inspection of all performed analysis.
Currently, this works for the face analysis only.

### Kinematics and optical flow extraction
runMF(dlc_dir) runs across the resulting marker files and automatically saves data frames including, among others, the following data:
* Pupil diameter (extracted based on circular fit of 6 pupil markers)
* Eye opening (based on distance of upper and lower eyelids)
* Face regions (automatically segmented based on facial landmarks extracted from markers)
  
  _Automatic face segmentation on In-distribution data_
  
  <img src="img/faceregions.gif" alt="Sample image" width="300"/>
  
  _Automatic face segmentation on Out-of-distribution data from collaborators and the publicly available [IBL Brainwide Map](https://www.internationalbrainlab.com/data) dataset_
  
  <img width="300" height="300" src="https://github.com/user-attachments/assets/4f089a92-83c5-498d-a76d-e55a8ddba071"/>

* Motion energy for each face region
* Optical flow angle and magnitude for each face region (extracted using Farneback dense optical flow)
* Whisking & sniffing frequency and phase
* Paw movement, stride frequency, gait information (based on kinematics of paw markers)
* Paw and tail angles

## Dependencies
This software relies heavily on [DeepLabCut](https://github.com/DeepLabCut/DeepLabCut/), [LightningPose](https://github.com/paninski-lab/lightning-pose), [RAFT](https://github.com/princeton-vl/RAFT) and [OpenCV](https://opencv.org/) libraries for its functionality.

## Contributors
MouseFlow is under active development and we welcome community contributions. We welcome your feedbacks and ideas. Please get in touch :) 

We thank our collaborators: Janelle Pakan (LIN Magdeburg), Emilie Macé (UMG Göttingen), Simon Musall (RWTH Aachen), Jan Gründemann (DZNE Bonn), Yangfan Peng (Charité Berlin), Ricardo Paricio-Montesinos (DZNE Bonn), Sanja Mikulovic (LIN Magdeburg), Petra Mocellin (LIN Magdeburg), Liudmila Sosulina (LIN Magdeburg), and Falko Fuhrmann (DZNE Bonn) for generously sharing their data, which allowed us to enhance the performance of our pre-trained models.

We are also greatful to Nick del Grosso (iBOTS iBehave Bonn) and Sangeetha Nandakumar (iBOTS iBehave Bonn) for their invaluable support and expertise in Python, especially during the early stage of setting up the package.
