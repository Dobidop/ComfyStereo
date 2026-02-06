import torch
import numpy as np
from typing import Union, List
from PIL import Image
import cv2
import gc

# Set to True to enable memory debugging output
DEBUG_MEMORY = False

def log_memory(label=""):
    """Log current memory usage for debugging (only when DEBUG_MEMORY is True)"""
    if not DEBUG_MEMORY:
        return
    import psutil
    process = psutil.Process()
    ram_mb = process.memory_info().rss / 1024 / 1024
    if torch.cuda.is_available():
        vram_mb = torch.cuda.memory_allocated() / 1024 / 1024
        vram_reserved = torch.cuda.memory_reserved() / 1024 / 1024
        print(f"[MEM] {label}: RAM={ram_mb:.0f}MB, VRAM={vram_mb:.0f}MB (reserved={vram_reserved:.0f}MB)")
    else:
        print(f"[MEM] {label}: RAM={ram_mb:.0f}MB")

from . import stereoimage_generation as sig

from comfy.utils import ProgressBar




def tensor2np(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor[0]  # Assuming we take the first image in the batch
    np_array = tensor.cpu().numpy()
    np_array = np.clip(255.0 * np_array, 0, 255).astype(np.uint8)
    if np_array.shape[0] == 3:  # Convert from (3, H, W) to (H, W, 3)
        np_array = np_array.transpose(1, 2, 0)
    return np_array
    
def np2tensor(img_np: Union[np.ndarray, List[np.ndarray]]) -> torch.Tensor:
    if isinstance(img_np, list):
        return torch.cat([np2tensor(img) for img in img_np], dim=0)
    return torch.from_numpy(img_np.astype(np.float32) / 255.0).unsqueeze(0)

class StereoImageNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "depth_map": ("IMAGE",),
                "modes": (["left-right", "right-left", "top-bottom", "bottom-top", "red-cyan-anaglyph"],),
                "fill_technique": ([
                     'No fill', 'No fill - Reverse projection', 'Imperfect fill - Hybrid Edge', 'Fill - Naive',
                    'Fill - Naive interpolating', 'Fill - Polylines Soft', 'Fill - Polylines Sharp', 'GPU Warp (Fast)'#,
                    #'Fill - Post-fill', 'Fill - Reverse projection with Post-fill', 'Fill - Hybrid Edge with fill'
                ], {"default": "Fill - Polylines Soft"}),
            },
            "optional": {
                "divergence": ("FLOAT", {"default": 3.5, "min": 0.05, "max": 15, "step": 0.01}),
                "separation": ("FLOAT", {"default": 0, "min": -5, "max": 5, "step": 0.01}),
                "stereo_balance": ("FLOAT", {"default": 0, "min": -0.95, "max": 0.95, "step": 0.05}),
                "convergence_point": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "stereo_offset_exponent": ("FLOAT", {"default": 2, "min": 1, "max": 2, "step": 1}),
                "depth_map_blur": ("BOOLEAN", {"default": True}),
                "depth_blur_edge_threshold": ("FLOAT", {"default": 6, "min": 0.1, "max": 15, "step": 0.1}),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32, "step": 1, "tooltip": "Number of frames to process before clearing GPU memory. Lower values use less memory but may be slower."}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("stereoscope", "blurred_depthmap_left", "blurred_depthmap_right", "no_fill_imperfect_mask")
    FUNCTION = "generate"

    def generate(self, image, depth_map, divergence, separation, modes,
                 stereo_balance, convergence_point, stereo_offset_exponent, fill_technique, depth_blur_edge_threshold, depth_map_blur, batch_size=4):

        log_memory("START of generate()")
        if DEBUG_MEMORY:
            print(f"[DEBUG] Processing {len(image)} frames with batch_size={batch_size}")
            print(f"[DEBUG] Input image shape: {image.shape}, depth_map shape: {depth_map.shape}")
            print(f"[DEBUG] Input image dtype: {image.dtype}, depth_map dtype: {depth_map.dtype}")

        fill_technique_mapping = {
            'GPU Warp (Fast)': 'gpu_warp',
            'No fill': 'none',
            'No fill - Reverse projection': 'inverse',
            'Imperfect fill - Hybrid Edge': 'hybrid_edge',
            'Fill - Naive': 'naive',
            'Fill - Naive interpolating': 'naive_interpolating',
            'Fill - Polylines Soft': 'polylines_soft',
            'Fill - Polylines Sharp': 'polylines_sharp',
            'Fill - Post-fill': 'none_post',
            'Fill - Reverse projection with Post-fill': 'inverse_post',
            'Fill - Hybrid Edge with fill': 'hybrid_edge_plus'
        }

        fill_technique = fill_technique_mapping.get(fill_technique, 'gpu_warp')

        # Use chunked concatenation to avoid memory spikes at the end
        # Instead of accumulating all frames then concatenating, we concatenate in chunks
        results_chunks = []
        depthmap_left_chunks = []
        depthmap_right_chunks = []
        mask_chunks = []

        # Temporary lists for current chunk
        results_batch = []
        depthmap_left_batch = []
        depthmap_right_batch = []
        mask_batch = []

        total_steps = len(image)
        pbar = ProgressBar(total_steps)

        log_memory("Before processing loop")

        # Process in batches to manage memory for large video sequences
        frames_since_cleanup = 0

        for i in range(len(image)):
            if i % 10 == 0:  # Log every 10 frames to avoid too much output
                log_memory(f"Frame {i}/{total_steps}")
            # ComfyUI uses [H, W, C] format, convert to [C, H, W] for processing
            img_tensor = image[i]  # [H, W, C]
            dm_tensor = depth_map[i]  # [H, W, C]

            # Convert from [H, W, C] to [C, H, W] format
            if img_tensor.dim() == 3 and img_tensor.shape[2] in [1, 3]:
                img_tensor = img_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            elif img_tensor.dim() == 2:
                img_tensor = img_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]

            if dm_tensor.dim() == 3 and dm_tensor.shape[2] in [1, 3]:
                dm_tensor = dm_tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]
            elif dm_tensor.dim() == 2:
                dm_tensor = dm_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]

            # Convert RGB depth map to grayscale if needed (now in [C, H, W] format)
            if dm_tensor.shape[0] == 3:
                dm_tensor = 0.2989 * dm_tensor[0] + 0.5870 * dm_tensor[1] + 0.1140 * dm_tensor[2]
            elif dm_tensor.shape[0] == 1:
                dm_tensor = dm_tensor.squeeze(0)

            # Ensure dm_tensor is 2D at this point
            if dm_tensor.dim() > 2:
                dm_tensor = dm_tensor.squeeze()

            # Resize depth map if needed
            if img_tensor.shape[1:] != dm_tensor.shape:
                dm_tensor = torch.nn.functional.interpolate(
                    dm_tensor.unsqueeze(0).unsqueeze(0),
                    size=img_tensor.shape[1:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze()

            # Use divergence as depth_blur_strength for directional motion blur
            depth_blur_strength = divergence if depth_map_blur else 0

            # Use fully GPU-accelerated path for gpu_warp technique
            if fill_technique == 'gpu_warp':
                results_tensors, left_depth, right_depth = sig.create_stereoimages_gpu(
                    img_tensor, dm_tensor, divergence, separation,
                    [modes], stereo_balance, stereo_offset_exponent, convergence_point,
                    depth_blur_strength, depth_blur_edge_threshold, depth_map_blur
                )
                # Convert GPU tensors to the expected format
                # Results are [C, H, W] tensors, need to convert to PIL-compatible format
                for result_tensor in results_tensors:
                    # Convert from [C, H, W] to [H, W, C] and scale to 0-255
                    result_np = result_tensor.permute(1, 2, 0).cpu().numpy()
                    result_np = (np.clip(result_np, 0, 1) * 255).astype(np.uint8)
                    results_batch.append(np2tensor(result_np))

                # Convert depth maps
                left_depth_np = left_depth.cpu().numpy() if left_depth.is_cuda else left_depth.numpy()
                right_depth_np = right_depth.cpu().numpy() if right_depth.is_cuda else right_depth.numpy()
                left_depth_np = (np.clip(left_depth_np, 0, 1) * 255).astype(np.uint8)
                right_depth_np = (np.clip(right_depth_np, 0, 1) * 255).astype(np.uint8)

                # Convert grayscale to RGB for consistency
                if left_depth_np.ndim == 2:
                    left_depth_np = np.stack([left_depth_np] * 3, axis=-1)
                if right_depth_np.ndim == 2:
                    right_depth_np = np.stack([right_depth_np] * 3, axis=-1)

                depthmap_left_batch.append(np2tensor(left_depth_np))
                depthmap_right_batch.append(np2tensor(right_depth_np))

                # Generate mask (for GPU warp, there are no unfilled pixels due to border padding)
                # Use original single-eye dimensions (correct for all modes: SBS, TB, anaglyph)
                mask = torch.zeros((1, img_tensor.shape[1], img_tensor.shape[2]))
                mask_batch.append(mask)

                # Memory management for large batches - concatenate and clear
                frames_since_cleanup += 1
                if frames_since_cleanup >= batch_size:
                    log_memory(f"Before batch concat (frame {i})")
                    # Concatenate current batch into a single tensor and store
                    if results_batch:
                        results_chunks.append(torch.cat(results_batch, dim=0))
                        depthmap_left_chunks.append(torch.cat(depthmap_left_batch, dim=0))
                        depthmap_right_chunks.append(torch.cat(depthmap_right_batch, dim=0))
                        mask_chunks.append(torch.cat(mask_batch, dim=0))

                        # Clear batch lists to free memory
                        results_batch = []
                        depthmap_left_batch = []
                        depthmap_right_batch = []
                        mask_batch = []

                    log_memory(f"After batch concat, before cache clear (frame {i})")
                    # Clear GPU cache to prevent memory buildup
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()  # Also trigger Python garbage collection
                    log_memory(f"After cache clear (frame {i})")
                    frames_since_cleanup = 0

                pbar.update(1)
                continue

            # CPU path for other fill techniques
            output = sig.create_stereoimages(img_tensor, dm_tensor, divergence, separation,
                                             [modes], stereo_balance, stereo_offset_exponent,
                                             fill_technique, depth_blur_strength, depth_blur_edge_threshold,
                                             depth_map_blur, convergence_point=convergence_point)

            
            if len(output) == 3:
                results, left_modified_depthmap, right_modified_depthmap = output
            else:
                results, modified_depthmap = output
                left_modified_depthmap = modified_depthmap
                right_modified_depthmap = modified_depthmap
            
            results_batch.append(convertResult(results))
            depthmap_left_batch.append(convertResult(left_modified_depthmap))
            depthmap_right_batch.append(convertResult(right_modified_depthmap))

            mask = self.generate_mask(results)
            mask_batch.append(mask)

            # Memory management for large batches (CPU path) - concatenate and clear
            frames_since_cleanup += 1
            if frames_since_cleanup >= batch_size:
                log_memory(f"CPU path: Before batch concat (frame {i})")
                # Concatenate current batch into a single tensor and store
                if results_batch:
                    results_chunks.append(torch.cat(results_batch, dim=0))
                    depthmap_left_chunks.append(torch.cat(depthmap_left_batch, dim=0))
                    depthmap_right_chunks.append(torch.cat(depthmap_right_batch, dim=0))
                    mask_chunks.append(torch.cat(mask_batch, dim=0))

                    # Clear batch lists to free memory
                    results_batch = []
                    depthmap_left_batch = []
                    depthmap_right_batch = []
                    mask_batch = []

                log_memory(f"CPU path: After batch concat (frame {i})")
                # Clear GPU cache if available (may have been used for depth map operations)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                log_memory(f"CPU path: After cache clear (frame {i})")
                frames_since_cleanup = 0

            pbar.update(1)

        # Handle any remaining frames in the last incomplete batch
        log_memory("After processing loop, before final batch handling")
        if results_batch:
            log_memory(f"Final batch: {len(results_batch)} remaining frames")
            results_chunks.append(torch.cat(results_batch, dim=0))
            depthmap_left_chunks.append(torch.cat(depthmap_left_batch, dim=0))
            depthmap_right_chunks.append(torch.cat(depthmap_right_batch, dim=0))
            mask_chunks.append(torch.cat(mask_batch, dim=0))
            # Clear to free memory before final concat
            results_batch = []
            depthmap_left_batch = []
            depthmap_right_batch = []
            mask_batch = []
            gc.collect()

        log_memory(f"Before final concat: {len(results_chunks)} chunks")
        if DEBUG_MEMORY:
            print(f"[DEBUG] Chunk sizes: {[c.shape for c in results_chunks]}")

        # Final memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        log_memory("After cleanup, before final assembly")

        # Pre-allocate final tensors and copy chunks in-place to avoid memory spike
        # This avoids torch.cat() which needs to hold both old chunks and new tensor simultaneously

        # Calculate total frames from chunks
        total_frames = sum(chunk.shape[0] for chunk in results_chunks)

        # Get shapes from first chunk
        _, h_result, w_result, c_result = results_chunks[0].shape
        _, h_depth, w_depth, c_depth = depthmap_left_chunks[0].shape
        _, h_mask, w_mask = mask_chunks[0].shape

        if DEBUG_MEMORY:
            print(f"[DEBUG] Pre-allocating tensors for {total_frames} frames")
        log_memory("Before pre-allocation")

        # Pre-allocate final tensors
        final_results = torch.empty((total_frames, h_result, w_result, c_result), dtype=results_chunks[0].dtype)
        log_memory("After results pre-alloc")

        final_depthmap_left = torch.empty((total_frames, h_depth, w_depth, c_depth), dtype=depthmap_left_chunks[0].dtype)
        log_memory("After depthmap_left pre-alloc")

        final_depthmap_right = torch.empty((total_frames, h_depth, w_depth, c_depth), dtype=depthmap_right_chunks[0].dtype)
        log_memory("After depthmap_right pre-alloc")

        final_mask = torch.empty((total_frames, h_mask, w_mask), dtype=mask_chunks[0].dtype)
        log_memory("After mask pre-alloc")

        # Copy chunks into pre-allocated tensors and free each chunk immediately
        current_idx = 0
        for i in range(len(results_chunks)):
            chunk_size = results_chunks[i].shape[0]
            end_idx = current_idx + chunk_size

            # Copy and immediately delete each chunk to free memory
            final_results[current_idx:end_idx] = results_chunks[i]
            results_chunks[i] = None  # Release reference

            final_depthmap_left[current_idx:end_idx] = depthmap_left_chunks[i]
            depthmap_left_chunks[i] = None

            final_depthmap_right[current_idx:end_idx] = depthmap_right_chunks[i]
            depthmap_right_chunks[i] = None

            final_mask[current_idx:end_idx] = mask_chunks[i]
            mask_chunks[i] = None

            current_idx = end_idx

            # Periodically trigger garbage collection
            if i % 10 == 0:
                gc.collect()
                if i % 20 == 0:
                    log_memory(f"Copied {i+1}/{len(results_chunks)} chunks")

        # Final cleanup
        results_chunks = []
        depthmap_left_chunks = []
        depthmap_right_chunks = []
        mask_chunks = []
        gc.collect()

        log_memory("END of generate() - returning results")
        if DEBUG_MEMORY:
            print(f"[DEBUG] Final output shapes: results={final_results.shape}, left={final_depthmap_left.shape}, right={final_depthmap_right.shape}, mask={final_mask.shape}")

        return (
            final_results,
            final_depthmap_left,
            final_depthmap_right,
            final_mask
        )

    def generate_mask(self, stereoscope_img):
        # Handle list of images (from CPU path) - take the first one
        if isinstance(stereoscope_img, list):
            stereoscope_img = stereoscope_img[0]
        np_img = np.array(stereoscope_img)
        mask = (np_img.sum(axis=-1) == 0).astype(np.uint8) * 255  # Black areas become white in mask
        return np2tensor(mask)



def convertResult(results):
    # Convert the result to a tensor
    if isinstance(results, list):
        results = np.array(results[0])
    else:
        results = np.array(results)

    # Ensure the results are in the correct shape (H, W, C)
    if len(results.shape) == 2:  # Convert grayscale to RGB
        results = np.stack([results]*3, axis=-1)

    results_tensor = np2tensor(results)
    
    return results_tensor
    

def tensor2cv2(image: torch.Tensor) -> List[tuple[np.ndarray, tuple]]:
    original_shape = image.shape
    images = []

    if image.dim() == 4:  # Batch of images
        for i in range(image.shape[0]):
            img = image[i]
            if img.shape[0] == 3:  # If in [C, H, W] format
                img = img.permute(1, 2, 0)
            npimage = (img * 255).byte().cpu().numpy()
            if npimage.shape[-1] == 3:
                images.append((cv2.cvtColor(npimage, cv2.COLOR_RGB2BGR), original_shape))
            elif npimage.shape[-1] == 1:
                images.append((npimage.squeeze(-1), original_shape))  # For grayscale images
            else:
                images.append((npimage, original_shape))  # Return as-is for other channel configurations
    else:  # Single image
        if image.shape[0] == 3:  # If in [C, H, W] format
            image = image.permute(1, 2, 0)
        npimage = (image * 255).byte().cpu().numpy()
        if npimage.shape[-1] == 3:
            images.append((cv2.cvtColor(npimage, cv2.COLOR_RGB2BGR), original_shape))
        elif npimage.shape[-1] == 1:
            images.append((npimage.squeeze(-1), original_shape))  # For grayscale images
        else:
            images.append((npimage, original_shape))  # Return as-is for other channel configurations

    return images

def cv22tensor(cv2_imgs: List[tuple[np.ndarray, tuple]]) -> torch.Tensor:
    tensors = []
    for cv2_img, original_shape in cv2_imgs:
        if cv2_img.ndim == 2:  # grayscale
            tensor = torch.from_numpy(cv2_img).float() / 255.0
            tensor = tensor.unsqueeze(-1)  # Add channel dimension
        else:  # color
            tensor = torch.from_numpy(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)).float() / 255.0

        if len(original_shape) == 4:
            tensor = tensor.permute(2, 0, 1).unsqueeze(0)  # [H, W, C] -> [1, C, H, W]
        elif len(original_shape) == 3 and original_shape[0] == 3:
            tensor = tensor.permute(2, 0, 1)  # [H, W, C] -> [C, H, W]

        tensors.append(tensor)

    return torch.stack(tensors)

def merge_channels(red_channel, green_channel, blue_channel):
    if red_channel.dim() == 4:  # Batch of images
        images = []
        for i in range(red_channel.shape[0]):
            image = mix_rgb_channels(tensor2pil(red_channel[i]).convert('L'), tensor2pil(
                green_channel[i]).convert('L'), tensor2pil(blue_channel[i]).convert('L'))
            images.append(pil2tensor(image).squeeze(0))  # Remove batch dimension added by pil2tensor
        return (torch.stack(images),)
    else:  # Single image
        image = mix_rgb_channels(tensor2pil(red_channel).convert('L'), tensor2pil(
            green_channel).convert('L'), tensor2pil(blue_channel).convert('L'))
        return (pil2tensor(image),)

def mix_rgb_channels(red, green, blue):
    # Create an empty image with the same size as the channels
    width, height = red.size
    merged_img = Image.new('RGB', (width, height))

    # Merge the channels into the new image
    merged_img = Image.merge('RGB', (red, green, blue))

    return merged_img

# PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


NODE_CLASS_MAPPINGS = {
    "StereoImageNode": StereoImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoImageNode": "Stereo Image Node",
}
