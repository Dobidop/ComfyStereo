import torch
import numpy as np
import os
import sys
from typing import Union, List
from PIL import Image
import cv2
import numpy as np
import torch
from typing import Union, List
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageDraw, ImageChops, ImageFont

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

import stereoimage_generation as sig

from comfy.utils import ProgressBar

import socket
import json
import io
import base64

import os
import socket
import json
import hashlib
import shutil

import subprocess
import psutil
import time

class DeoVRLauncher:
    def __init__(self, config_path):
        self.config_path = config_path
        self.deovr_path = self.load_deovr_path()

    def load_deovr_path(self):
        """Loads the DeoVR installation path from a config file."""
        if not os.path.exists(self.config_path):
            print(f"❌ Config file not found: {self.config_path}")
            return None

        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
                return config.get("deovr_path", None)
        except Exception as e:
            print(f"❌ Error reading config file: {e}")
            return None

    def is_deovr_running(self):
        """Checks if DeoVR.exe is currently running."""
        for process in psutil.process_iter(attrs=['name']):
            if process.info['name'].lower() == "deovr.exe":
                print("✅ DeoVR is already running.")
                return True
        print("ℹ️ DeoVR is not running.")
        return False

    def create_dummy_image(self, temp_folder):
        """Creates a black dummy image in the temp folder."""
        dummy_path = os.path.join(temp_folder, "dummy.png")

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        try:
            img = Image.new("RGB", (1920, 1080), (0, 0, 0))  # Black image
            img.save(dummy_path)
            print(f"✅ Created dummy image: {dummy_path}")
            return dummy_path
        except Exception as e:
            print(f"❌ Error creating dummy image: {e}")
            return None

    def launch_deovr(self):
        """Launches DeoVR with the dummy image if it's not already running."""
        if self.is_deovr_running():
            print("⚠️ DeoVR is already running. No need to launch.")
            return

        if not self.deovr_path:
            print("❌ DeoVR path is not set. Check the config file.")
            return

        script_directory = os.path.dirname(os.path.abspath(__file__))
        loading_image_path = os.path.join(script_directory, 'loading_sbs_flat.png')

        try:
            subprocess.Popen([self.deovr_path, loading_image_path], shell=True)
            print(f"🚀 Launched DeoVR...")
            print(f"Waiting for DeoVR to start...")
            time.sleep(5)
            print(f"Still waiting for DeoVR to start... heat death of the universe approaching...")
            time.sleep(5)
            print(f"DeoVR... Any moment now... Soon™")
            time.sleep(5)
        except Exception as e:
            print(f"❌ Error launching DeoVR: {e}")


class DeoVRViewNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": ("STRING", {"forceInput": True}),  # User-provided file or video path
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
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "send_path_to_deovr"
    CATEGORY = "Output"
    OUTPUT_NODE = True

    def send_path_to_deovr(self, file_path, projection_type, eye_location):
        """
        Copies the file to ComfyUI's temp folder with a modified name based on the projection type and eye location.
        A unique MD5 hash is computed for the file to ensure uniqueness. If the file has been copied before,
        the existing copy is reused. The command is then sent to DeoVR with the new file path.
        """
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "./config.json"))
        launcher = DeoVRLauncher(config_path)
        launcher.launch_deovr()
        
        
        if not os.path.exists(file_path):
            print(f"Error: The specified file does not exist: {file_path}")
            return ()

        # Define the temp folder relative to ComfyUI (two levels up)
        temp_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../temp"))

        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # Compute a unique hash for the file (MD5 is fast for most files)
        file_hash = self.compute_file_hash(file_path)

        # Get the base name and extension of the file
        base_name = os.path.basename(file_path)
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
                shutil.copy2(file_path, new_file_path)
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

        # Send the command.
        self.send_command_to_deovr(command)
        return ()

    def send_command_to_deovr(self, command):
        """Handles sending the command to the DeoVR remote control API."""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.connect(("127.0.0.1", 23554))  # DeoVR's remote control port
                command_json = json.dumps(command)
                length_prefix = len(command_json).to_bytes(4, byteorder='little')
                packet = length_prefix + command_json.encode('utf-8')
                s.sendall(packet)
                print(f"✅ Sent command to DeoVR: {command}")
        except Exception as e:
            print(f"❌ Error sending command to DeoVR: {e}")

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
                    'No fill', 'No fill - Reverse projection', 'Imperfect fill - Hybrid Edge', 'Fill - Naive',
                    'Fill - Naive interpolating', 'Fill - Polylines Soft', 'Fill - Polylines Sharp',
                    'Fill - Post-fill', 'Fill - Reverse projection with Post-fill', 'Fill - Hybrid Edge with fill'
                ], {"default": "Fill - Polylines Soft"}),
                #"return_basic_mask": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "divergence": ("FLOAT", {"default": 3.5, "min": 0.05, "max": 15, "step": 0.01}), 
                "separation": ("FLOAT", {"default": 0, "min": -5, "max": 5, "step": 0.01}),
                "stereo_balance": ("FLOAT", {"default": 0, "min": -0.95, "max": 0.95, "step": 0.05}),
                "stereo_offset_exponent": ("FLOAT", {"default": 2, "min": 1, "max": 2, "step": 1}),
                "depth_blur_sigma": ("FLOAT", {"default": 0, "min": 0, "max": 10, "step": 0.1}),
                "depth_blur_edge_threshold": ("FLOAT", {"default": 40, "min": 0.1, "max": 100, "step": 0.1})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK")
    RETURN_NAMES = ("stereoscope", "modified_depthmap_left", "modified_depthmap_right", "no_fill_imperfect_mask")
    FUNCTION = "generate"

    def generate(self, image, depth_map, divergence, separation, modes, 
                 stereo_balance, stereo_offset_exponent, fill_technique, depth_blur_sigma, depth_blur_edge_threshold):# return_basic_mask):
        
        fill_technique_mapping = {
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
        
        fill_technique = fill_technique_mapping.get(fill_technique, 'none')
          
        results_final = []
        modified_depthmap_final_left = []
        modified_depthmap_final_right = []
        mask_final = []
        total_steps = len(image)
        pbar = ProgressBar(total_steps)
        
        for i in range(len(image)):
            img = tensor2np(image[i:i+1])
            dm = tensor2np(depth_map[i:i+1])
        
            if len(dm.shape) == 3 and dm.shape[2] == 3:
                dm = np.dot(dm[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            if img.shape[:2] != dm.shape:
                dm = np.array(Image.fromarray(dm).resize((img.shape[1], img.shape[0])))
            
            output = sig.create_stereoimages(img, dm, divergence, separation,  
                                             [modes], stereo_balance, stereo_offset_exponent, 
                                             fill_technique, depth_blur_sigma, depth_blur_edge_threshold)
            
            if len(output) == 3:
                results, left_modified_depthmap, right_modified_depthmap = output
            else:
                results, modified_depthmap = output
                left_modified_depthmap = modified_depthmap
                right_modified_depthmap = modified_depthmap
            
            results_final.append(convertResult(results))
            modified_depthmap_final_left.append(convertResult(left_modified_depthmap))
            modified_depthmap_final_right.append(convertResult(right_modified_depthmap))
            
            mask = self.generate_mask(results)
            mask_final.append(mask)
            
            pbar.update(1)
        
        return (torch.cat(results_final), torch.cat(modified_depthmap_final_left), torch.cat(modified_depthmap_final_right), torch.cat(mask_final))

    def generate_mask(self, stereoscope_img):
        np_img = np.array(stereoscope_img)
        mask = (np_img.sum(axis=-1) == 0).astype(np.uint8) * 255  # Black areas become white in mask
        return np2tensor(mask)



def convertResult(results):
    # Save the result for debugging
    if isinstance(results, list):
        for idx, result in enumerate(results):
            result = np.array(result)
            #Image.fromarray(result).save(f"debug_result_{idx}.png")
    else:
        results = np.array(results)
        #Image.fromarray(results).save("debug_result.png")

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


# class LazyStereo:
    # @classmethod
    # def INPUT_TYPES(s):
        # return {"required": {
            # "image": ("IMAGE",),
            # "depth_map": ("IMAGE",),
            # "shift_amount": ("INT", {
                # "default": 10, 
                # "min": 5, #Minimum value
                # "max": 200, #Maximum value
                # "step": 1, #Slider's step
                # "display": "number" # Cosmetic only: display as "number" or "slider"
            # }),
            # "mode": (["Cross-eyed", "Parallel"],),
            # },
        # }

    # RETURN_TYPES = ("IMAGE",)
    # FUNCTION = "generate_cross_eyed_image"

    # def generate_cross_eyed_image(self, image, depth_map, shift_amount=10, mode="Cross-eyed"):
        # images = tensor2cv2(image)
        # depth_maps = tensor2cv2(depth_map)

        # results = []
        # total_steps = len(image)  # Total number of images to process
        # pbar = ProgressBar(total_steps)

        # for (img, original_img_shape), (depth_map, original_depth_shape) in zip(images, depth_maps):
            # if len(depth_map.shape) == 3 and depth_map.shape[2] > 1:
                # depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
            # elif len(depth_map.shape) == 2:
                # # It's already grayscale, no need to convert
                # pass
            # else:
                # raise ValueError(f"Unexpected depth map shape: {depth_map.shape}")

            # if img is None or depth_map is None:
                # raise ValueError("Image or depth map not found.")

            # height, width = img.shape[:2]

            # # Calculate gradients to determine high-contrast regions
            # grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
            # grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
            # gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            # max_gradient = np.max(gradient_magnitude)
            # gradient_magnitude = gradient_magnitude / max_gradient  # Normalize to [0, 1]

            # # Initialize the left and right images and masks for inpainting
            # left_image = np.zeros_like(img)
            # right_image = np.zeros_like(img)
            # left_mask = np.zeros((height, width), dtype=np.uint8)
            # right_mask = np.zeros((height, width), dtype=np.uint8)

            # # Generate the left and right images by shifting pixels according to the depth map
            # for y in range(height):
                # for x in range(width):
                    # depth = depth_map[y, x]
                    # local_gradient = gradient_magnitude[y, x]
                    # adaptive_shift = shift_amount * (1 - local_gradient)  # Reduce shift in high-contrast areas
                    # shift = int((depth / 255.0) * adaptive_shift)
                    # if x - shift >= 0:
                        # left_image[y, x] = img[y, x - shift]
                    # else:
                        # left_image[y, x] = img[y, 0]
                    # if x + shift < width:
                        # right_image[y, x] = img[y, x + shift]
                    # else:
                        # right_image[y, x] = img[y, width - 1]

                    # # Update masks for inpainting
                    # if x - shift < 0 or x + shift >= width:
                        # left_mask[y, x] = 255
                        # right_mask[y, x] = 255

            # # Apply edge detection to create a more precise inpainting mask
            # edges = cv2.Canny(depth_map, 100, 200)
            # left_mask = cv2.bitwise_or(left_mask, edges)
            # right_mask = cv2.bitwise_or(right_mask, edges)

            # # Inpaint the left and right images to fill in gaps
            # left_image = cv2.inpaint(left_image, left_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            # right_image = cv2.inpaint(right_image, right_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            # if mode == "Cross-eyed":
                # # Combine the left and right images side by side
                # cross_eyed_image = np.hstack((right_image, left_image))
            # else:
                # cross_eyed_image = np.hstack((left_image, right_image))

            # # Separate the channels
            # red_channel = cross_eyed_image[:, :, 0]
            # green_channel = cross_eyed_image[:, :, 1]
            # blue_channel = cross_eyed_image[:, :, 2]

            # # Convert channels back to tensors
            # red_tensor = cv22tensor([(red_channel, (1, 1, height, width))])
            # green_tensor = cv22tensor([(green_channel, (1, 1, height, width))])
            # blue_tensor = cv22tensor([(blue_channel, (1, 1, height, width))])

            # merged_image = merge_channels(blue_tensor, green_tensor, red_tensor)

            # results.append(merged_image[0])
            # pbar.update(1)

        # # Concatenate the results into a single tensor
        # return (torch.cat(results),)

NODE_CLASS_MAPPINGS = {
#    "LazyStereo": LazyStereo,
    "StereoImageNode": StereoImageNode,
    "DeoVRViewNode": DeoVRViewNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
#    "LazyStereo": "LazyStereo",
    "StereoImageNode": "Stereo Image Node",
    "DeoVRViewNode": "DeoVR View"
}
