import torch
import numpy as np
import os
import sys
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

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import stereoimage_generation as sig

from comfy.utils import ProgressBar

import socket
import hashlib
import shutil
import time
import json
import psutil
import subprocess




class DeoVRLauncher:
    def __init__(self, config_path):
        self.config_path = config_path
        self.deovr_path = self.load_deovr_path()

    def load_deovr_path(self):
        """Loads the DeoVR installation path from a config file."""
        if not os.path.exists(self.config_path):
            print(f"X  Config file not found: {self.config_path}")
            return None

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                return config.get("deovr_path", None)
        except Exception as e:
            print(f"X  Error reading config file: {e}")
            return None

    def is_process_running(self, process_name):
        """Checks if a process with the given name is running."""
        for process in psutil.process_iter(attrs=['name']):
            if process.info['name'].lower() == process_name.lower():
                return True
        return False

    def is_deovr_running(self):
        """Checks if DeoVR.exe is currently running."""
        return self.is_process_running("deovr.exe")

    def create_dummy_image(self, temp_folder):
        """Creates a black dummy image in the temp folder."""
        dummy_path = os.path.join(temp_folder, "dummy.png")

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        try:
            img = Image.new("RGB", (1920, 1080), (0, 0, 0))  # Black image
            img.save(dummy_path)
            print(f"OK!  Created dummy image: {dummy_path}")
            return dummy_path
        except Exception as e:
            print(f"X  Error creating dummy image: {e}")
            return None

    def launch_deovr(self):
        """Launches DeoVR and waits for it to start, checking every 2 seconds for up to 30 seconds."""

        if self.is_deovr_running():
            print("OK! DeoVR is already running. No need to launch.")
            return

        if not self.deovr_path:
            print("XX  DeoVR path is not set. Check the config file.")
            return
            
        if not os.path.exists(self.deovr_path):
            print("\nXXX  DeoVR path seems to be incorrect, file not found:")
            print(f"XXX  {self.deovr_path}")
            print("XXX  Update the config file to point to your DeoVR.exe file to autostart it.")
            print("!!!  You can still use this manually by launching DeoVR and opening an image/video.")
            return

        # Check if VR-related processes are running
        vr_processes = ["vrmonitor.exe", "vrwebhelper.exe"]
        initial_wait_time = 0

        for proc in vr_processes:
            if not self.is_process_running(proc):
                print(f"! {proc} is not running. Waiting an extra 3 seconds.")
                initial_wait_time += 3

        # Try launching DeoVR
        script_directory = os.path.dirname(os.path.abspath(__file__))
        loading_image_path = os.path.join(script_directory, 'loading_sbs_flat.png')

        try:
            subprocess.Popen([self.deovr_path, loading_image_path], shell=True)
            print(">> Launched DeoVR...")
        except Exception as e:
            print(f"XXX  Error launching DeoVR: {e}")
            return

        # Wait for DeoVR to start (polling every 2 seconds, max 30 seconds + extra time)
        max_wait_time = 30 + initial_wait_time
        elapsed_time = 1

        print(" Waiting for DeoVR to start...")

        time.sleep(initial_wait_time)

        while elapsed_time < max_wait_time:
            if self.is_deovr_running():
                print("OK! DeoVR has successfully launched.")
                #time.sleep(10)  # Final wait before proceeding
                return

            if elapsed_time % 6 == 0:  # Print status every 6 seconds
                print(f" Still waiting... ({elapsed_time}/{max_wait_time} seconds)")
            time.sleep(2)
            elapsed_time += 2

        print("XXX  ERROR: DeoVR did not start within the expected time.")



class DeoVRViewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_name": ("STRING", {"forceInput": True}),  # User-provided file or video path
                "projection_type": ([
                    "Flat 2D plane",
                    "Equirectangular - 180° FOV",
                    "Equirectangular - 360° FOV",
                    "Fisheye projection - 180° FOV",
                    "Fisheye projection - 190° FOV",
                    "Fisheye projection - 200° FOV with MKX lens correction",
                    "Fisheye projection - 220° FOV with VRCA lens correction"
                ],),
                "eye_location": (["Side-by-Side", "Top-Bottom"],),
                "file_location": (["Output folder", "Input folder", "Other (provide full path)"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_path_to_deovr"
    CATEGORY = "Output"
    OUTPUT_NODE = True

    def send_path_to_deovr(self, file_name, projection_type, eye_location, file_location):
        """
        Copies the file to ComfyUI's temp folder with a modified name based on the projection type and eye location.
        A unique MD5 hash is computed for the file to ensure uniqueness. If the file has been copied before,
        the existing copy is reused. The command is then sent to DeoVR with the new file path.
        """
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./config.json"))
        launcher = DeoVRLauncher(config_path)
        launcher.launch_deovr()
        
        if file_location == "Output folder":
            file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../output/", file_name))

        if file_location == "Input folder":
            file_name = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../input/", file_name))


        if not os.path.exists(file_name):
            print(f"Error: The specified file does not exist: {file_name}")
            return ()

        # Define the temp folder relative to ComfyUI (two levels up)
        temp_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../temp"))

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # Compute a unique hash for the file (MD5 is fast for most files)
        file_hash = self.compute_file_hash(file_name)

        # Get the base name and extension of the file
        base_name = os.path.basename(file_name)
        base, ext = os.path.splitext(base_name)

        # Define suffix mappings (inspired by generateJson.py)
        projection_suffix_mapping = {
            "Flat 2D plane": "_screen",
            "Equirectangular - 180° FOV": "_180",
            "Equirectangular - 360° FOV": "_360",
            "Fisheye projection - 180° FOV": "_fisheye",
            "Fisheye projection - 190° FOV": "_fisheye190",
            "Fisheye projection - 200° FOV with MKX lens correction": "_mkx200",
            "Fisheye projection - 220° FOV with VRCA lens correction": "_vrca220"
        }
        stereo_suffix_mapping = {
            "Side-by-Side": "_SBS",
            "Top-Bottom": "_TB"
        }

        proj_suffix = projection_suffix_mapping.get(projection_type, "")
        stereo_suffix = stereo_suffix_mapping.get(eye_location, "")

        # Create the new file name: [hash]_[base][proj_suffix][stereo_suffix][ext]
        new_file_name = f"{file_hash}_{base}{proj_suffix}{stereo_suffix}{ext}"
        new_file_path = os.path.join(temp_folder, new_file_name)

        # If the file doesn't already exist in the temp folder, copy it.
        if not os.path.exists(new_file_path):
            try:
                shutil.copy2(file_name, new_file_path)
                print(f"Copied file to: {new_file_path}")
            except Exception as e:
                print(f"Error copying file: {e}")
                return ()

        # Create the JSON command to send to DeoVR.
        command = {
            "path": new_file_path,
            "screenType": proj_suffix[1:] if proj_suffix.startswith("_") else "flat",
            "stereoMode": stereo_suffix[1:].lower() if stereo_suffix.startswith("_") else "sbs",
            "is3d": True
        }


        max_retries = 10
        tries = 0
        # Send the command
        while tries <= max_retries:
            message, errormessage = self.send_command_to_deovr(command)

            if message == 'error':
                #print(f"Retrying to send command to DeoVR")
                time.sleep(2)
                tries = tries + 1
            else:
                return ()
            
        print(f"Failed to send command to DeoVR")
        print(f"Error message: {errormessage}")
        return ()

    def send_command_to_deovr(self, command):
        """Handles sending a command to the DeoVR remote control API and returns the response."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", 23554))  # DeoVR's remote control port
                command_json = json.dumps(command)
                length_prefix = len(command_json).to_bytes(4, byteorder='little')
                packet = length_prefix + command_json.encode('utf-8')
                
                # Send command
                s.sendall(packet)
                #print(f"✅ Sent command to DeoVR: {command}")

                # Read response from server
                response_length_bytes = s.recv(4)  # First 4 bytes indicate message length
                if not response_length_bytes:
                    #print("❌ No response received from DeoVR.")
                    return None
                
                response_length = int.from_bytes(response_length_bytes, byteorder='little')
                response_data = s.recv(response_length).decode('utf-8')
                response_json = json.loads(response_data)

                #print(f"📩 Received response from DeoVR: {response_json}")
                return response_json, ''  # Return the parsed response

        except Exception as e:
            #print(f"❌ Error sending command to DeoVR: {e}")
            return 'error', e

    def compute_file_hash(self, file_path):
        """Computes an MD5 hash of the file for a unique identifier."""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except Exception as e:
            print(f"Error computing file hash: {e}")
            return "errorhash"



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
                    'GPU Warp (Fast)', 'No fill', 'No fill - Reverse projection', 'Imperfect fill - Hybrid Edge', 'Fill - Naive',
                    'Fill - Naive interpolating', 'Fill - Polylines Soft', 'Fill - Polylines Sharp'#,
                    #'Fill - Post-fill', 'Fill - Reverse projection with Post-fill', 'Fill - Hybrid Edge with fill'
                ], {"default": "GPU Warp (Fast)"}),
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
                mask = torch.zeros((1, results_tensors[0].shape[1], results_tensors[0].shape[2] // 2))
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
    "DeoVRViewNode": DeoVRViewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StereoImageNode": "Stereo Image Node",
    "DeoVRViewNode": "DeoVR View"
}
