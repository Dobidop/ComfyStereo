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
                "fill_technique": (['none', 'naive', 'naive_interpolating', 'polylines_soft','polylines_sharp'],),
            },
            "optional": {
                "divergence": ("FLOAT", {"default": 2.5, "min": 0.05, "max": 15, "step": 0.01}), 
                "separation": ("FLOAT", {"default": 0, "min": -5, "max": 5, "step": 0.01}),
                "stereo_balance": ("FLOAT", {"default": 0, "min": -0.95, "max": 0.95, "step": 0.05}),
                "stereo_offset_exponent": ("FLOAT", {"default": 2, "min": 1, "max": 2, "step": 1})
            }
        }
    
    RETURN_TYPES = ("IMAGE",) 
    FUNCTION = "generate"

    def generate(self, image, depth_map, divergence, separation, modes, 
                 stereo_balance, stereo_offset_exponent, fill_technique):
        
        results_final = []
        for i in range(len(image)):
            img = tensor2np(image[i:i+1])
            dm = tensor2np(depth_map[i:i+1])

        
            # Ensure depth_map is a single-channel grayscale image
            if len(dm.shape) == 3 and dm.shape[2] == 3:
                dm = np.dot(dm[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)
            
            # Ensure both images are of the same size
            if img.shape[:2] != dm.shape:
                dm = np.array(Image.fromarray(dm).resize((img.shape[1], img.shape[0])))

            results = sig.create_stereoimages(img, dm, divergence, separation,  
                                               [modes], stereo_balance, stereo_offset_exponent, fill_technique)

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
            
            results_final.append(results_tensor)

        return (torch.cat(results_final),)




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

class LazyStereo:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "image": ("IMAGE",),
            "depth_map": ("IMAGE",),
            "shift_amount": ("INT", {
                "default": 10, 
                "min": 5, #Minimum value
                "max": 200, #Maximum value
                "step": 1, #Slider's step
                "display": "number" # Cosmetic only: display as "number" or "slider"
            }),
            "mode": (["crosseye", "straight"],),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_cross_eyed_image"

    def generate_cross_eyed_image(self, image, depth_map, shift_amount=10, mode="crosseye"):
        images = tensor2cv2(image)
        depth_maps = tensor2cv2(depth_map)

        results = []

        for (img, original_img_shape), (depth_map, original_depth_shape) in zip(images, depth_maps):
            if len(depth_map.shape) == 3 and depth_map.shape[2] > 1:
                depth_map = cv2.cvtColor(depth_map, cv2.COLOR_BGR2GRAY)
            elif len(depth_map.shape) == 2:
                # It's already grayscale, no need to convert
                pass
            else:
                raise ValueError(f"Unexpected depth map shape: {depth_map.shape}")

            if img is None or depth_map is None:
                raise ValueError("Image or depth map not found.")

            height, width = img.shape[:2]

            # Calculate gradients to determine high-contrast regions
            grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            max_gradient = np.max(gradient_magnitude)
            gradient_magnitude = gradient_magnitude / max_gradient  # Normalize to [0, 1]

            # Initialize the left and right images and masks for inpainting
            left_image = np.zeros_like(img)
            right_image = np.zeros_like(img)
            left_mask = np.zeros((height, width), dtype=np.uint8)
            right_mask = np.zeros((height, width), dtype=np.uint8)

            # Generate the left and right images by shifting pixels according to the depth map
            for y in range(height):
                for x in range(width):
                    depth = depth_map[y, x]
                    local_gradient = gradient_magnitude[y, x]
                    adaptive_shift = shift_amount * (1 - local_gradient)  # Reduce shift in high-contrast areas
                    shift = int((depth / 255.0) * adaptive_shift)
                    if x - shift >= 0:
                        left_image[y, x] = img[y, x - shift]
                    else:
                        left_image[y, x] = img[y, 0]
                    if x + shift < width:
                        right_image[y, x] = img[y, x + shift]
                    else:
                        right_image[y, x] = img[y, width - 1]

                    # Update masks for inpainting
                    if x - shift < 0 or x + shift >= width:
                        left_mask[y, x] = 255
                        right_mask[y, x] = 255

            # Apply edge detection to create a more precise inpainting mask
            edges = cv2.Canny(depth_map, 100, 200)
            left_mask = cv2.bitwise_or(left_mask, edges)
            right_mask = cv2.bitwise_or(right_mask, edges)

            # Inpaint the left and right images to fill in gaps
            left_image = cv2.inpaint(left_image, left_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
            right_image = cv2.inpaint(right_image, right_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

            if mode == "crosseye":
                # Combine the left and right images side by side
                cross_eyed_image = np.hstack((right_image, left_image))
            else:
                cross_eyed_image = np.hstack((left_image, right_image))

            # Separate the channels
            red_channel = cross_eyed_image[:, :, 0]
            green_channel = cross_eyed_image[:, :, 1]
            blue_channel = cross_eyed_image[:, :, 2]

            # Convert channels back to tensors
            red_tensor = cv22tensor([(red_channel, (1, 1, height, width))])
            green_tensor = cv22tensor([(green_channel, (1, 1, height, width))])
            blue_tensor = cv22tensor([(blue_channel, (1, 1, height, width))])

            merged_image = merge_channels(blue_tensor, green_tensor, red_tensor)

            results.append(merged_image[0])

        # Concatenate the results into a single tensor
        return (torch.cat(results),)

NODE_CLASS_MAPPINGS = {
    "LazyStereo": LazyStereo,
    "StereoImageNode": StereoImageNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LazyStereo": "LazyStereo",
    "StereoImageNode": "Stereo Image Node"
}
