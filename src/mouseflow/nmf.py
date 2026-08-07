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
import matplotlib as mpl

from sklearn.decomposition import NMF
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
        
        self.spatial_scale = 2
        self.h_down = int(self.height // self.spatial_scale)
        self.w_down = int(self.width // self.spatial_scale)
        self.probe_sigma = self.width * 0.15
        self.guardrail_sigma = self.width * 0.15
        self.scaled_keypoints = [
            [x/self.spatial_scale, y/self.spatial_scale]
            for x, y in self.keypoints
        ]
        
        self.nmf_model = self._configure_nmf()
        self.nmf_spatial_components = None

    def get_frame_idxs(self, video_path, n_calib_frames, frame_step=1):
        """
        Randomly samples frame indices from a video for NMF calibration.
        Ensures that the sampled frames are not too close to the end of the video.
        Returns:
            List of frame indices to sample from the video.
        """
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

        if valid_max < n_calib_frames:
            return list(range(valid_max))

        # Sample random frame indices across the whole video
        random_idxs = random.sample(range(valid_max), n_calib_frames)
        
        return random_idxs
    
    def _create_spatial_priors(self, n_calib_frames, radius=90):
        """
        Creates spatial priors for NMF based on anatomical keypoints.
        Returns:
            H_init: Initial spatial components (n_components x n_features) - priored
            W_init: Initial temporal components (n_calib_frames x n_components) - not priored
        """
        n_channels = self.n_channels 
        n_features = n_channels * self.h_down * self.w_down
        
        H_init = np.full((self.n_components, n_features), 1e-7, dtype=np.float32)
        
        for i, (x, y) in enumerate(self.scaled_keypoints):
            if np.isnan(x) or np.isnan(y):
                continue
            mask = np.zeros((self.h_down, self.w_down), dtype=np.float32)
            
            cv2.circle(mask, (int(x), int(y)), radius, 1.0, thickness=-1)
            mask = cv2.GaussianBlur(mask, (31, 31), 0) 
            
            flattened_prior = np.tile(mask.ravel(), n_channels)
            
            H_init[i] = np.maximum(H_init[i], flattened_prior)
        
        W_init = (np.random.rand(n_calib_frames , self.n_components) * 0.1)
            
        return H_init.astype(np.float32), W_init.astype(np.float32)

    def _configure_nmf(self):
        return NMF(
            n_components=self.n_components, 
            init='custom',
            solver='mu',
            max_iter=400,
            tol=1e-3,
            alpha_H=0.6,
            l1_ratio=1.0
        )
        
    def fit(self, flow_buffer):
        """
        Fits the NMF model to a given buffer of optical flow frames
        priored on keypoints.
        The flow_buffer should be a list of 2D arrays (one per channel) for each frame.
        """
        X_fit = np.array(flow_buffer, dtype=np.float32)
        H_init, W_init = self._create_spatial_priors(n_calib_frames=300, radius=90)
        max_val = np.max(X_fit)
        if max_val > 0:
            X_fit /= max_val
        self.nmf_model.fit_transform(X_fit, H=H_init, W=W_init)
        print(f"NMF fit completed with {self.nmf_model.n_iter_} iterations.")
        raw_components = self.nmf_model.components_
        
        # Upscale components back to full resolution, if they were downsampled for fitting (e.g, for speed reasons)
        self.nmf_spatial_components = np.empty((self.n_components, self.n_channels * self.height * self.width), dtype=np.float32)
        for i in range(self.n_components):
            comp_2d = raw_components[i].reshape(self.n_channels, self.h_down, self.w_down)
            resized_components = []
            for c in range(self.n_channels):
                resized = cv2.resize(comp_2d[c], (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                resized_components.append(resized)
            self.nmf_spatial_components[i] = np.concatenate([comp.ravel() for comp in resized_components])
            
    def get_dynamic_masks(self, cur_gray, video_path):
        """
        Matches the NMF components back to anatomical keypoints and builds dynamic masks.
        Returns:
            masks_sp: Sparse matrix of the new masks (each row corresponds to a mask)
            px_counts: Array of pixel counts for each mask (used for averaging flow values later)
        """
        matched_indices, label_map = self._match_components_to_keypoints()
        masks_sp, px_counts = self._build_anatomical_islands(
            matched_indices, label_map, cur_gray, video_path
        )
        return masks_sp, px_counts
    
    def _match_components_to_keypoints(self):
        """
        Matches the NMF components back to anatomical keypoints based on spatial overlap.
        Returns:
            matched_indices: List of matched NMF component indices for each keypoint (-1 if no match)
            label_map: Dictionary mapping matched NMF component indices to anatomical labels
        """
        print('Matching NMF components back to anatomical keypoints...')
        num_anchors = min(len(self.keypoints), len(self.base_masks))
        
        norm_footprints = []
        for nmf_idx in range(self.n_components):
            region_1d = self.nmf_spatial_components[nmf_idx]
            region_3d = region_1d.reshape(self.n_channels, self.height, self.width)
            raw_footprint = np.sum(region_3d, axis=0)
            total_mass = np.sum(raw_footprint)
            if total_mass > 0:
                norm_footprints.append(raw_footprint / total_mass)
            else:
                norm_footprints.append(raw_footprint)

        overlap_matrix = np.zeros((num_anchors, self.n_components), dtype=np.float32)
        
        for r_idx in range(num_anchors): # FIX: Dynamic range based on num_anchors
            cx, cy = self.keypoints[r_idx]
            if np.isnan(cx) or np.isnan(cy):
                overlap_matrix[r_idx, :] = -1.0
                continue
            y_idx, x_idx = np.indices((self.height, self.width))
            probe = np.exp(-((x_idx - cx)**2 + (y_idx - cy)**2) / (2 * self.probe_sigma**2))
            
            for nmf_idx in range(self.n_components):
                overlap_matrix[r_idx, nmf_idx] = np.sum(norm_footprints[nmf_idx] * probe)

        matched_indices = [-1] * num_anchors
        MIN_OVERLAP_THRESHOLD = 0.005 

        for _ in range(num_anchors): # FIX: Dynamic range
            best_r, best_c = np.unravel_index(np.argmax(overlap_matrix), overlap_matrix.shape)
            best_score = overlap_matrix[best_r, best_c]
            print(f"Best match: Keypoint {best_r} to Component {best_c} with score {best_score:.4f}")
            if best_score < MIN_OVERLAP_THRESHOLD:
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
        """
        Builds anatomical masks based on matched NMF components and keypoints.
        Returns:
            masks_sp: Sparse matrix of the new masks (each row corresponds to a mask)
            px_counts: Array of pixel counts for each mask (used for averaging flow values later)
        """
        new_masks = []
        new_px_counts = []
        num_anchors = min(len(self.keypoints), len(self.base_masks))
        plot_masks_2d = {}
        
        for r_idx in range(num_anchors): #looping trhough all anchor keypoints
            true_nmf_idx = matched_indices[r_idx]
            cx, cy = self.keypoints[r_idx]
            if np.isnan(cx) or np.isnan(cy):
                true_nmf_idx = -1
            
            if true_nmf_idx != -1:
                region_1d = self.nmf_spatial_components[true_nmf_idx]
                region_3d = region_1d.reshape(self.n_channels, self.height, self.width)
                raw_footprint = np.sum(region_3d, axis=0)
            
                cx_int, cy_int = int(round(self.keypoints[r_idx][0])), int(round(self.keypoints[r_idx][1]))
                y_idx, x_idx = np.indices((self.height, self.width))
                guardrail = np.exp(-((x_idx - cx)**2 + (y_idx - cy)**2) / (2 * self.guardrail_sigma**2))
                
                clean_footprint = raw_footprint * guardrail
                
                max_w = np.max(clean_footprint)
                if max_w > 0:
                    clean_footprint /= max_w
                
                binary_mask = clean_footprint > 0.3

                labeled_array, num_features = ndi.label(binary_mask)
                blob_id = labeled_array[cy_int, cx_int]
                
                if blob_id > 0:
                    anatomical_mask = (labeled_array == blob_id).astype(np.float32)
                else:
                    anatomical_mask = (clean_footprint > 0.3).astype(np.float32)
                    
                plot_masks_2d[true_nmf_idx] = anatomical_mask
                flat_weights = anatomical_mask.ravel()
                new_masks.append(flat_weights)
                new_px_counts.append(np.sum(flat_weights) + 1e-9)
            else:
                mask = self.base_masks[r_idx]
                if sp.issparse(mask):
                    mask = mask.toarray()
                orig_mask_flat = mask.ravel().astype(np.float32)
                new_masks.append(orig_mask_flat)
                new_px_counts.append(self.px_per_mask[r_idx] + 1e-9)

        for r_idx in range(num_anchors, len(self.base_masks)):
            mask = self.base_masks[r_idx]
            if sp.issparse(mask):
                mask = mask.toarray()
            orig_mask_flat = mask.ravel().astype(np.float32)
            new_masks.append(orig_mask_flat)
            new_px_counts.append(self.px_per_mask[r_idx] + 1e-9)
        
        self.save_mask_overlays(label_map, cur_gray, video_path, final_masks=plot_masks_2d)
        masks_sp = sp.csr_matrix(np.vstack(new_masks))
        px_counts = np.array(new_px_counts) 
        
        return masks_sp, px_counts
    
    
    ##============================Just helpers for plotting and saving the masks============================##
    def save_mask_overlays(self, label_map, cur_gray, video_path, final_masks=None):
        fig, axes = plt.subplots(4, 3, figsize=(15, 16))
        axes = axes.flatten()
        out_dir = video_path.parent / "mouseflow"
        out_dir.mkdir(exist_ok=True, parents=True)
        out_path = out_dir / f"{video_path.stem}_nmf_masks.png"

        for nmf_idx in range(self.n_components):
            ax = axes[nmf_idx]
            region = self.nmf_spatial_components[nmf_idx].reshape(self.n_channels, self.height, self.width)
            footprint = np.sum(region, axis=0)
            ax.imshow(cur_gray, cmap='gray')
            max_val = np.max(footprint)
            if max_val > 0:
                masked_footprint = np.ma.masked_where(footprint < 0.05 * max_val, footprint)
                ax.imshow(footprint, cmap='magma', alpha=0.6)
            
            if final_masks and nmf_idx in final_masks:
                mask_2d = final_masks[nmf_idx]
                ax.contour(mask_2d > 0, levels=[0.5], colors='cyan', linewidths=2)
            
            if nmf_idx in label_map:
                ax.set_title(f"Comp {nmf_idx}: {label_map[nmf_idx]}", color='red', fontweight='bold')
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(3)
            else:
                ax.set_title(f"Comp {nmf_idx}: Background")
                ax.axis('off')
                
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
        
        mpl.rcParams['figure.dpi'] = 150
        mpl.rcParams['pdf.fonttype'] = 42
        mpl.rcParams['ps.fonttype'] = 42

        target_labels = ['Whiskerpad', 'Nose'] 
        target_indices = [
            idx for idx, label in label_map.items() 
            if any(target in label for target in target_labels)
        ]

        if target_indices:
            fig_clean, ax_clean = plt.subplots(figsize=(10, 8))
            ax_clean.imshow(cur_gray, cmap='gray', rasterized=True)

            cmap_dict = {
                'Whiskerpad': plt.cm.magma,
                'Nose': plt.cm.viridis,
                'Mouth': plt.cm.plasma,
                'Chin': plt.cm.winter
            }
            default_cmap = plt.cm.Purples

            master_footprint = np.zeros((self.height, self.width), dtype=np.float32)

            for nmf_idx in target_indices:
                region = self.nmf_spatial_components[nmf_idx].reshape(self.n_channels, self.height, self.width)
                footprint = np.sum(region, axis=0)
                max_val = np.max(footprint)
                
                if max_val == 0:
                    continue

                master_footprint = np.maximum(master_footprint, footprint)

                smoothed_surface = ndi.gaussian_filter(footprint, sigma=10)
                smoothed_max = np.max(smoothed_surface)
                
                if smoothed_max == 0:
                    continue

                norm_surface = smoothed_surface / smoothed_max
                
                region_name = label_map[nmf_idx]
                cmap = default_cmap
                for name, cm in cmap_dict.items():
                    if name in region_name:
                        cmap = cm
                        break
                
                rgba_img = cmap(norm_surface)
                alpha_channel = (norm_surface ** 1.5) * 0.85 
                rgba_img[..., 3] = alpha_channel
                
                ax_clean.imshow(rgba_img, rasterized=True)

            global_max = np.max(master_footprint)
            if global_max > 0:
                global_smoothed = ndi.gaussian_filter(master_footprint, sigma=10)
                global_smoothed_max = np.max(global_smoothed)
                contour_levels = np.linspace(global_smoothed_max * 0.25, global_smoothed_max * 0.90, 4)
                
                ax_clean.contour(
                    global_smoothed, 
                    levels=contour_levels, 
                    colors='white', 
                    linewidths=1.2, 
                    alpha=0.7
                )

            ax_clean.axis('off')
            clean_out_path = out_dir / f"{video_path.stem}_combined_kde_map.png"
            plt.savefig(clean_out_path, dpi=300, bbox_inches='tight', pad_inches=0)

            svg_out_path = out_dir / f"{video_path.stem}_combined_kde_map.pdf"
            plt.savefig(svg_out_path, format='pdf', bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
            plt.close(fig_clean)
            
            if target_indices and final_masks:
                for nmf_idx in target_indices:
                    fig_binary, ax_binary = plt.subplots(figsize=(10, 8))
                    ax_binary.imshow(cur_gray, cmap='gray', rasterized=True)

                    region = self.nmf_spatial_components[nmf_idx].reshape(self.n_channels, self.height, self.width)
                    footprint = np.sum(region, axis=0)
                    max_val = np.max(footprint)
                    
                    if max_val == 0:
                        plt.close(fig_binary)
                        continue

                    smoothed_surface = ndi.gaussian_filter(footprint, sigma=10)
                    smoothed_max = np.max(smoothed_surface)
                    
                    if smoothed_max == 0:
                        plt.close(fig_binary)
                        continue

                    norm_surface = smoothed_surface / smoothed_max
                    
                    region_name = label_map[nmf_idx]
                    cmap = default_cmap
                    for name, cm in cmap_dict.items():
                        if name in region_name:
                            cmap = cm
                            break
                    
                    rgba_img = cmap(norm_surface)
                    alpha_channel = (norm_surface ** 1.5) * 0.85 
                    rgba_img[..., 3] = alpha_channel
                    
                    ax_binary.imshow(rgba_img, rasterized=True)

                    if nmf_idx in final_masks:
                        mask_2d = final_masks[nmf_idx]
                        ax_binary.contour(
                            mask_2d > 0, 
                            levels=[0.5], 
                            colors='white', 
                            linewidths=2,
                            alpha=0.9
                        )
                        

                    ax_binary.axis('off')
                    ax_binary.scatter(self.keypoints[0][0], self.keypoints[0][1], 
                                      color='cyan', edgecolors='black', s=50, zorder=5)
                    ax_binary.scatter(self.keypoints[1][0], self.keypoints[1][1], 
                                      color='magenta', edgecolors='black', s=50, zorder=5)
                    
                    safe_region_name = region_name.replace(" ", "_").lower()
                    binary_pdf_path = out_dir / f"{video_path.stem}_{safe_region_name}_binary_map.pdf"
                    
                    plt.savefig(binary_pdf_path, format='pdf', bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
                    plt.close(fig_binary)

            if target_indices and final_masks:
                fig_binary, ax_binary = plt.subplots(figsize=(10, 8))
                ax_binary.imshow(cur_gray, cmap='gray', rasterized=True)

                for nmf_idx in target_indices:
                    region = self.nmf_spatial_components[nmf_idx].reshape(self.n_channels, self.height, self.width)
                    footprint = np.sum(region, axis=0)
                    max_val = np.max(footprint)
                    
                    if max_val == 0:
                        continue

                    smoothed_surface = ndi.gaussian_filter(footprint, sigma=10)
                    smoothed_max = np.max(smoothed_surface)
                    
                    if smoothed_max == 0:
                        continue

                    norm_surface = smoothed_surface / smoothed_max
                    
                    region_name = label_map[nmf_idx]
                    cmap = default_cmap
                    for name, cm in cmap_dict.items():
                        if name in region_name:
                            cmap = cm
                            break
                    
                    rgba_img = cmap(norm_surface)
                    alpha_channel = (norm_surface ** 1.5) * 0.85 
                    rgba_img[..., 3] = alpha_channel
                    
                    ax_binary.imshow(rgba_img, rasterized=True)

                    if nmf_idx in final_masks:
                        mask_2d = final_masks[nmf_idx]
                        ax_binary.contour(
                            mask_2d > 0, 
                            levels=[0.5], 
                            colors='white', 
                            linewidths=2,
                            alpha=0.9
                        )

                ax_binary.axis('off')
                binary_pdf_path = out_dir / f"{video_path.stem}_combined_binary_map.pdf"
                plt.savefig(binary_pdf_path, format='pdf', bbox_inches='tight', pad_inches=0, transparent=True, dpi=300)
                plt.close(fig_binary)