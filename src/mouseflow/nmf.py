"""
Author: @Jakob Faust
Date: 07.08.2026


NMF wrapper class for creating and fitting Non-negative Matrix Factorization models to optical flow data.
The fitting process is priored with spatial priors based on anatomical keypoints.
"""

import numpy as np
import cv2
import scipy.ndimage as ndi
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from pathlib import Path

import random

class NMFBuilder:
    def __init__(self, width, height, base_masks, px_per_mask, keypoints, n_components=12, down_factor=2, n_channels=2):
        self.width = width
        self.height = height
        self.base_masks = base_masks
        self.px_per_mask = px_per_mask
        self.n_components = n_components
        self.keypoints = keypoints
        self.down_factor = down_factor
        self.n_channels = n_channels
        
        self.spatial_scale = down_factor
        self.h_down = int(self.height // self.spatial_scale)
        self.w_down = int(self.width // self.spatial_scale)
        
        distances = []
        for i in range(len(self.keypoints)):
            for j in range(i + 1, len(self.keypoints)):
                if not np.isnan(self.keypoints[i][0]) and not np.isnan(self.keypoints[j][0]):
                    dist = np.linalg.norm(np.array(self.keypoints[i]) - np.array(self.keypoints[j]))
                    distances.append(dist)

        min_kp_dist = np.min(distances) if distances else self.width * 0.10
        # Constrain Gaussian spread to half the distance to the nearest neighboring structure
        adaptive_sigma = max(min_kp_dist * 0.40, 5.0) 

        self.probe_sigma = adaptive_sigma
        self.guardrail_sigma = adaptive_sigma
        self.scaled_keypoints = [
            [x/self.spatial_scale, y/self.spatial_scale]
            for x, y in self.keypoints
        ]
        
        self.nmf_model = self._configure_nmf()
        self.nmf_spatial_components = None

    def get_frame_idxs(self, video_path, n_calib_frames, frame_step=1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print("Error: Could not open video.")
            return []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Ensure we don't pick a frame so close to the end that we can't form a pair
        valid_max = frame_count - frame_step - 1 
        
        if valid_max <= 0:
            return []

        # If the video is shorter than the requested frames, just take what we have
        if valid_max < n_calib_frames:
            return list(range(valid_max))

        # Sample completely random frame indices across the whole video
        random_idxs = random.sample(range(valid_max), n_calib_frames)
        return random_idxs

    
    
    def _create_nmf_priors(self, n_calib_frames, radius=60, X_flow=None):
        '''
        Creates spatial priors for NMF initilisation process.

        '''
        n_channels = self.n_channels 
        n_features = n_channels * self.h_down * self.w_down
        
        H_init = np.full((self.n_components, n_features), 1e-7, dtype=np.float32)
        
        # Build spatial priors (H_init)
        for i, (x, y) in enumerate(self.scaled_keypoints): # loop through all anatomical landmarks
            if np.isnan(x) or np.isnan(y):
                continue

            # init empty prior in frame size
            mask = np.zeros((self.h_down, self.w_down), dtype=np.float32)
            
            # init a kernel for gaussian blur (odd size)
            kernel_size = int(self.probe_sigma * 4)
            if kernel_size % 2 == 0:
                kernel_size += 1
            kernel_size = max(kernel_size, 15)

            # place a filled circle (1.o) of given radius on the anatomical landmark
            cv2.circle(mask, (int(x), int(y)), int(self.probe_sigma * 1.5), 1.0, thickness=-1)
            # weight the circle filling and edges by gaussian blurring
            mask = cv2.GaussianBlur(mask, (kernel_size, kernel_size), sigmaX=self.probe_sigma)
            
            # flatten the gaussian prior and duplicate it across all K channels
            flattened_prior = np.tile(mask.ravel(), n_channels) # shape (K x n_channels)

            # init the nmf spatial matrix with the obtained prior.
            # np.maximum to ensure overlapping priors don't get replaced during iterations over landmarks
            H_init[i] = np.maximum(H_init[i], flattened_prior)
        
        # Build Temporal Priors (W_init)
        W_init = np.zeros((n_calib_frames, self.n_components), dtype=np.float32)
        
        if X_flow is not None:
            for i in range(self.n_components):
                mask_1d = H_init[i]
                
                # Identify pixels with meaningful spatial prior weighting
                active_pixels = mask_1d > np.max(mask_1d) * 0.1 
                
                if np.sum(active_pixels) > 0:
                    # Calculate the mean motion vector over time for this anatomical region
                    W_init[:, i] = np.mean(X_flow[:, active_pixels], axis=1)
                else:
                    # Fallback for unassigned background components
                    W_init[:, i] = np.random.rand(n_calib_frames) * 0.1
        else:
            W_init = np.random.rand(n_calib_frames, self.n_components) * 0.1
            
        # Add small noise layer to prevent zero-gradient trapping in the NMF mu solver
        W_init += (np.random.rand(*W_init.shape) * 0.01)
            
        return H_init.astype(np.float32), W_init.astype(np.float32)

    def _configure_nmf(self):
        return NMF(
            n_components=self.n_components, 
            init='custom',
            solver='mu',
            max_iter=400,
            tol=1e-3,
            alpha_H=0.9,
            l1_ratio=1.0
        )
        
    def fit(self, flow_buffer):
        '''
        Fits  NMF model to a given buffer of flow frames
        '''
        X_fit_full = np.array(flow_buffer, dtype=np.float32) # non-negative rectified flow matrix
        
        active_mask = np.any(X_fit_full > 0, axis=0) # currently not activated, but could be used to filter out noise
        X_fit_active = X_fit_full[:, active_mask]
        
        # normalize matrix to [0, 1]
        max_val = np.max(X_fit_active)
        if max_val > 0:
            X_fit_active /= max_val
            
        # Slice the spatial prior to match active pixels
        H_init_full, W_init = self._create_nmf_priors(
            n_calib_frames=X_fit_full.shape[0], radius=90, X_flow=X_fit_full)
        H_init_active = H_init_full[:, active_mask]

        # Fit on the active matrix
        self.nmf_model.fit_transform(X_fit_active, H=H_init_active, W=W_init)
        print(f"NMF fit completed on active pixels with {self.nmf_model.n_iter_} iterations.")
        
        # Reconstruct full-sized components array with zeros for the background
        raw_components_active = self.nmf_model.components_
        raw_components_full = np.zeros((self.n_components, X_fit_full.shape[1]), dtype=np.float32)
        
        # Map the active computed weights back to their original spatial positions
        for i in range(self.n_components):
            raw_components_full[i, active_mask] = raw_components_active[i]
        
        # Upscale components back to full resolution (in case frames were downsampled before)
        self.nmf_spatial_components = np.empty((self.n_components, self.n_channels * self.height * self.width), dtype=np.float32)
        for i in range(self.n_components):
            comp_2d = raw_components_full[i].reshape(self.n_channels, self.h_down, self.w_down)
            resized_components = []
            for c in range(self.n_channels):
                resized = cv2.resize(comp_2d[c], (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                resized_components.append(resized)
            self.nmf_spatial_components[i] = np.concatenate([comp.ravel() for comp in resized_components])
            
    def get_dynamic_masks(self, cur_gray, video_path):
        '''
        Returns final binary masks and there corrseponding pixel counts found via NMF
        '''
        matched_indices, label_map = self._match_components_to_keypoints()
        masks_sp, px_counts = self._build_anatomical_islands(
            matched_indices, label_map, cur_gray, video_path
        )
        return masks_sp, px_counts
    
    def _match_components_to_keypoints(self):
        '''
        After having found arbitrary NMF components, this function matches them back to actual anatomical landmarks.
        Via greedy maximum spatial overlap algorithm.
        '''
        print('Matching NMF components back to anatomical keypoints...')
        num_anchors = min(len(self.keypoints), len(self.base_masks))
        norm_footprints = []
        for nmf_idx in range(self.n_components): # loop through all nmf components
            region_1d = self.nmf_spatial_components[nmf_idx]
            region_3d = region_1d.reshape(self.n_channels, self.height, self.width) # channels are nmf channels (e.g. np.abs(u), nps.abs(v), etc.)
            raw_footprint = np.sum(region_3d, axis=0) # yields single 2D heatmap for this nmf component
            total_mass = np.sum(raw_footprint) # L1 density normalization
            if total_mass > 0:
                norm_footprints.append(raw_footprint / total_mass)
            else:
                norm_footprints.append(raw_footprint)

        # creates an empty overlap matrix
        overlap_matrix = np.zeros((num_anchors, self.n_components), dtype=np.float32)
        for r_idx in range(num_anchors):
            cx, cy = self.keypoints[r_idx]
            if np.isnan(cx) or np.isnan(cy):
                overlap_matrix[r_idx, :] = -1.0
                continue
            y_idx, x_idx = np.indices((self.height, self.width))
            probe = np.exp(-((x_idx - cx)**2 + (y_idx - cy)**2) / (2 * self.probe_sigma**2)) # place a gaussian on each landmark point
            
            for nmf_idx in range(self.n_components):
                overlap_matrix[r_idx, nmf_idx] = np.sum(norm_footprints[nmf_idx] * probe) # calculate overlap (inner product) with nmf compnent

        matched_indices = [-1] * num_anchors
        MIN_OVERLAP_THRESHOLD = 0.005 
        # greedy matching: for each landmark, find best matching nmf component, remove component from list for next landmarks
        # r: landmark idx
        # c: nmf component idx
        for _ in range(num_anchors):
            best_r, best_c = np.unravel_index(np.argmax(overlap_matrix), overlap_matrix.shape)
            best_score = overlap_matrix[best_r, best_c]
            if best_score < MIN_OVERLAP_THRESHOLD: # minimum overlap, otherwise just don't use nmf mask for this landmark
                break

            matched_indices[best_r] = best_c
            overlap_matrix[best_r, :] = -1.0
            overlap_matrix[:, best_c] = -1.0

        for r_idx, match in enumerate(matched_indices):
            if match != -1:
                print(f"  -> Keypoint {r_idx} successfully matched to NMF Component {match}")
            else:
                print(f"  -> Warning: Keypoint {r_idx} FAILED to match. Falling back to original mask.")

        # Ensure we only map keys that actually exist in the keypoints list
        base_labels = ["Nose", "Whiskerpad", "Mouth", "Chin"]
        label_map = {}
        for i in range(num_anchors):
            if matched_indices[i] != -1:
                label_map[matched_indices[i]] = base_labels[i]
        
        return matched_indices, label_map

    def _build_anatomical_islands(self, matched_indices, label_map, cur_gray, video_path):
        new_masks = []
        new_px_counts = []
        num_anchors = min(len(self.keypoints), len(self.base_masks))
        plot_masks_2d = {}
        
        for r_idx in range(num_anchors):
            true_nmf_idx = matched_indices[r_idx]
            cx, cy = self.keypoints[r_idx]
            if np.isnan(cx) or np.isnan(cy):
                true_nmf_idx = -1
            
            if true_nmf_idx != -1: # a nmf component was found for this keypoint
                region_1d = self.nmf_spatial_components[true_nmf_idx]
                region_3d = region_1d.reshape(self.n_channels, self.height, self.width)
                raw_footprint = np.sum(region_3d, axis=0) # 2D nmf heatmap 
            
                cx_int, cy_int = int(round(self.keypoints[r_idx][0])), int(round(self.keypoints[r_idx][1]))
                cx_int = max(0, min(cx_int, self.width - 1))
                cy_int = max(0, min(cy_int, self.height - 1))
                y_idx, x_idx = np.indices((self.height, self.width))
                guardrail = np.exp(-((x_idx - cx)**2 + (y_idx - cy)**2) / (2 * self.guardrail_sigma**2)) # place gaussian on landmark
                
                clean_footprint = raw_footprint * guardrail # multiply nmf component by gaussain to ensure anatomical plausibility
                
                # normalize nmf componennt to [0, 1]
                max_w = np.max(clean_footprint)
                if max_w > 0:
                    clean_footprint /= max_w
                
                footprint_8u = (clean_footprint * 255).astype(np.uint8) # integer representation of nmf mask
                nonzero_pixels = footprint_8u[footprint_8u > 5] # filter out 0 nmf weight componennts

                if len(nonzero_pixels) > 10:
                    # find the best possible threshold to differentiate background from actual foreground weights
                    otsu_thresh, _ = cv2.threshold(nonzero_pixels, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    # scale threshould back to [0, 1] + add a small penalty for large weights
                    dynamic_thresh = (otsu_thresh / 255.0) * 1.2
                    # clip threshold to exclude extreme values 
                    dynamic_thresh = np.clip(dynamic_thresh, 0.2, 0.8)
                else:
                    dynamic_thresh = 0.5

                # get back binary mask from nmf component heatmap
                binary_mask = clean_footprint > dynamic_thresh

                # groups adjacent ones in the binary pixel mask to continous 'islands' (groups) and assigns each group an individual id.
                # pixels belonging to no group (aka background noise) are assigned to group 0
                labeled_array, num_features = ndi.label(binary_mask)
                # get the group id at the position of the anatomical landmark
                blob_id = labeled_array[cy_int, cx_int]
                
                if blob_id > 0: # actually found a corresponding conected nmf component
                    anatomical_mask = (labeled_array == blob_id).astype(np.float32)
                else: # no continous nmf component found, just use arbitrary thresholding for binarization 
                    anatomical_mask = (clean_footprint > 0.5).astype(np.float32)
                    
                
                plot_masks_2d[true_nmf_idx] = anatomical_mask # save mask for plotting etc
                flat_weights = anatomical_mask.ravel() 
                new_masks.append(flat_weights) # save identified mask for downstream processing
                new_px_counts.append(np.sum(flat_weights) + 1e-9)
            
            else: # failed matching of nmf component or missing landmark, just use orifinal mask provided via landmark geometry
                mask = self.base_masks[r_idx]
                if sp.issparse(mask):
                    mask = mask.toarray()
                orig_mask_flat = mask.ravel().astype(np.float32)
                new_masks.append(orig_mask_flat)
                new_px_counts.append(self.px_per_mask[r_idx] + 1e-9)

        # loop through all remaining provided original masks and append them to the new masks list
        for r_idx in range(num_anchors, len(self.base_masks)):
            mask = self.base_masks[r_idx]
            if sp.issparse(mask):
                mask = mask.toarray()
            orig_mask_flat = mask.ravel().astype(np.float32)
            new_masks.append(orig_mask_flat)
            new_px_counts.append(self.px_per_mask[r_idx] + 1e-9)

        # save mask to files
        self.save_mask_overlays(label_map, cur_gray, video_path, final_masks=plot_masks_2d)

        # convert dense masks with many 0s to sparse masks for more efficient processing 
        masks_sp = sp.csr_matrix(np.vstack(new_masks))
        # pixel counts (mask size) for downstream normalisation etc
        px_counts = np.array(new_px_counts) 
        return masks_sp, px_counts
    
    def save_mask_overlays(self, label_map, cur_gray, video_path, final_masks=None):
        '''
        Saves mask picture as control output
        '''
        fig_all, ax_all = plt.subplots(figsize=(10, 8))
        ax_all.imshow(cur_gray, cmap='gray', rasterized=True)

        cmap_dict_local = {
            'Whiskerpad': plt.cm.magma,
            'Nose': plt.cm.viridis,
            'Mouth': plt.cm.plasma
        }
        default_cmap = plt.cm.Purples

        # lot all spatial masks as distinct colored overlays
        if final_masks:
            for nmf_idx, mask_2d in final_masks.items():
                if nmf_idx in label_map:
                    region_name = label_map[nmf_idx]
                    if region_name == 'Chin' or region_name == 'Cheek':
                        continue
                    cmap = default_cmap
                    for name, cm in cmap_dict_local.items():
                        if name in region_name:
                            cmap = cm
                            break
                    
                    edge_color = cmap(0.8)
                    
                    ax_all.contour(
                        mask_2d > 0, 
                        levels=[0.5], 
                        colors=[edge_color], 
                        linewidths=2.5,
                        alpha=0.9
                    )
                    
                    fill_overlay = np.zeros((*mask_2d.shape, 4))
                    fill_overlay[mask_2d > 0] = cmap(0.6) # Base color
                    fill_overlay[mask_2d > 0, 3] = 0.35   # 35% opacity
                    ax_all.imshow(fill_overlay)

        # Scatter plot all anatomical landmarks (Keypoints)
        if hasattr(self, 'keypoints') and self.keypoints is not None:
            for idx, (x, y) in enumerate(self.keypoints):
                if not (np.isnan(x) or np.isnan(y)): # Filter out dropped/missing DLC/SLEAP points
                    ax_all.scatter(
                        x, y, 
                        color='red', 
                        marker='+', 
                        s=150,           # Size of the marker
                        linewidths=2.5, 
                        zorder=5,        # Ensure markers render on top of the masks
                        label='Landmark' if idx == 0 else "" # Prevent legend duplication
                    )
                    

        ax_all.axis('off')
        video_path = Path(video_path)
        out_dir = video_path.parent / 'mouseflow'
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as a high-resolution PNG
        combined_png_path = out_dir / f"{video_path.stem}_all_masks_and_landmarks.png"
        plt.savefig(combined_png_path, format='png', bbox_inches='tight', pad_inches=0.1, transparent=True, dpi=300)
        print(f"Saved comprehensive mask/landmark validation PNG to: {combined_png_path}")
        plt.close(fig_all)