"""
Native VR viewer nodes for ComfyUI using PyOpenXR
Auto-launches directly into VR headset without browser
"""

import hashlib
import torch
import numpy as np
from PIL import Image
import os
import folder_paths

from .native_viewer import (
    check_openxr_available,
    launch_native_viewer,
    StereoFormat,
    PYOPENXR_AVAILABLE
)


class NativeStereoImageViewer:
    """
    Native VR stereo image viewer using PyOpenXR.
    Auto-launches directly into VR headset without browser.
    Requires OpenXR runtime (SteamVR, Oculus, WMR).
    """

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "output"
        self.prefix_append = "native_stereo_view"

    @classmethod
    def INPUT_TYPES(cls):
        # Check if OpenXR is available
        available, message = check_openxr_available()

        return {
            "required": {
                "image": ("IMAGE",),
                "stereo_format": (["Side-by-Side", "Over-Under", "Mono"],),
                "projection_type": (["Flat Screen", "Curved Screen", "180° Dome", "360° Sphere"],),
                "screen_size": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "screen_distance": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "swap_eyes": ("BOOLEAN", {"default": False}),
                "auto_launch": ("BOOLEAN", {"default": True}),
                "background_color": (["Black", "Dark Gray", "Gray", "White"],),
            },
            "optional": {
                "right_image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "view_stereo_native"
    OUTPUT_NODE = True
    CATEGORY = "stereo/native"

    def view_stereo_native(self, image, stereo_format, projection_type, screen_size, screen_distance, swap_eyes=False, auto_launch=True, background_color="Black", right_image=None):
        """
        View stereo image in VR headset using native PyOpenXR viewer.
        Auto-launches directly into headset.
        """
        # Check if PyOpenXR is available
        available, message = check_openxr_available()

        if not available:
            print(f"\n{'='*60}")
            print("NATIVE VR VIEWER NOT AVAILABLE")
            print(f"{'='*60}")
            print(f"Reason: {message}")
            print("\nTo enable native VR viewing:")
            print("1. Install PyOpenXR dependencies:")
            print("   pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw")
            print("2. Install a VR runtime:")
            print("   - SteamVR (recommended, supports most headsets)")
            print("   - Oculus Runtime (for Oculus headsets)")
            print("   - Windows Mixed Reality (built into Windows)")
            print("3. Connect your VR headset")
            print(f"{'='*60}\n")
            return (image,)

        # Convert tensor to numpy array
        if isinstance(image, torch.Tensor):
            img_np = (image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        else:
            img_np = image

        # Handle separate L/R images
        if right_image is not None:
            if isinstance(right_image, torch.Tensor):
                right_np = (right_image.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
            else:
                right_np = right_image

            # Create side-by-side layout from separate images
            if swap_eyes:
                sbs_img = np.concatenate([right_np, img_np], axis=1)
            else:
                sbs_img = np.concatenate([img_np, right_np], axis=1)

            img_np = sbs_img
            stereo_format = "Side-by-Side"

        # Save the stereo image
        img_hash = hashlib.md5(img_np.tobytes()).hexdigest()[:16]
        filename = f"{self.prefix_append}_{img_hash}.png"
        filepath = os.path.join(self.output_dir, filename)

        pil_img = Image.fromarray(img_np)
        pil_img.save(filepath)

        print(f"\n{'='*60}")
        print("NATIVE VR VIEWER")
        print(f"{'='*60}")
        print(f"Image saved: {filepath}")
        print(f"Stereo format: {stereo_format}")
        print(f"Swap eyes: {swap_eyes}")

        # Map format to internal format strings
        format_map = {
            "Side-by-Side": StereoFormat.SIDE_BY_SIDE,
            "Over-Under": StereoFormat.OVER_UNDER,
            "Mono": StereoFormat.MONO,
        }

        internal_format = format_map.get(stereo_format, StereoFormat.SIDE_BY_SIDE)

        # Map projection type to internal format
        projection_map = {
            "Flat Screen": "flat",
            "Curved Screen": "curved",
            "180° Dome": "dome180",
            "360° Sphere": "sphere360",
        }
        internal_projection = projection_map.get(projection_type, "flat")

        if auto_launch:
            print(f"{'='*60}\n")

            # Map background color to RGB tuple
            bg_color_map = {
                "Black": (0.0, 0.0, 0.0),
                "Dark Gray": (0.2, 0.2, 0.2),
                "Gray": (0.5, 0.5, 0.5),
                "White": (1.0, 1.0, 1.0),
            }
            bg_color = bg_color_map.get(background_color, (0.0, 0.0, 0.0))

            # Launch or update the persistent viewer
            # The viewer handles its own threading internally
            success = launch_native_viewer(
                filepath, internal_format, swap_eyes,
                projection_type=internal_projection,
                screen_size=screen_size,
                screen_distance=screen_distance,
                background_color=bg_color
            )

            if success:
                print("✓ VR viewer updated successfully")
                print("✓ Run the workflow again to update with new images")
            else:
                print("✗ Failed to launch VR viewer")
        else:
            print("\nAuto-launch disabled.")
            print("To manually launch, run:")
            print(f"python -m ComfyStereoViewer.native_viewer {filepath} {internal_format} {swap_eyes}")
            print(f"{'='*60}\n")

        # Return original image as passthrough
        return (image,)


class NativeVRStatus:
    """
    Check if native VR viewing is available and show status.
    Useful for debugging VR setup issues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("STRING", "BOOLEAN")
    RETURN_NAMES = ("status_message", "is_available")
    FUNCTION = "check_status"
    OUTPUT_NODE = True
    CATEGORY = "stereo/native"

    def check_status(self):
        """Check native VR viewer availability"""
        available, message = check_openxr_available()

        status_lines = [
            "="*60,
            "NATIVE VR VIEWER STATUS",
            "="*60,
        ]

        if available:
            status_lines.extend([
                "✓ PyOpenXR: Installed",
                "✓ OpenXR Runtime: Available",
                "✓ Status: Ready for VR viewing",
                "",
                "Detected runtime: " + message,
                "",
                "You can use native VR viewer nodes!",
            ])
        else:
            status_lines.extend([
                "✗ Status: Not available",
                "✗ Reason: " + message,
                "",
                "To enable native VR viewing:",
                "1. Install dependencies:",
                "   pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw",
                "2. Install a VR runtime:",
                "   - SteamVR (recommended)",
                "   - Oculus Runtime",
                "   - Windows Mixed Reality",
                "3. Connect your VR headset and start the runtime",
            ])

        status_lines.append("="*60)

        status_text = "\n".join(status_lines)
        print(status_text)

        return (status_text, available)


class NativeStereoVideoViewer:
    """
    Native VR stereo video viewer using PyOpenXR.
    Plays stereo videos directly in VR headset with keyboard controls.
    Requires OpenXR runtime (SteamVR, Oculus, WMR).
    """

    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {"default": "", "multiline": False}),
                "stereo_format": (["Side-by-Side", "Over-Under", "Mono"],),
                "projection_type": (["Flat Screen", "Curved Screen", "180° Dome", "360° Sphere"],),
                "screen_size": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "screen_distance": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.5}),
                "swap_eyes": ("BOOLEAN", {"default": False}),
                "loop_video": ("BOOLEAN", {"default": True}),
                "auto_launch": ("BOOLEAN", {"default": True}),
                "background_color": (["Black", "Dark Gray", "Gray", "White"],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "view_stereo_video"
    OUTPUT_NODE = True
    CATEGORY = "stereo/native"

    def view_stereo_video(self, video_path, stereo_format, projection_type, screen_size, screen_distance, swap_eyes=False, loop_video=True, auto_launch=True, background_color="Black"):
        """
        View stereo video in VR headset using native PyOpenXR viewer.
        Auto-launches directly into headset with video playback controls.
        """
        # Check if PyOpenXR is available
        available, message = check_openxr_available()

        if not available:
            print(f"\n{'='*60}")
            print("NATIVE VR VIEWER NOT AVAILABLE")
            print(f"{'='*60}")
            print(f"Reason: {message}")
            print("\nTo enable native VR viewing:")
            print("1. Install PyOpenXR dependencies:")
            print("   pip install pyopenxr PyOpenGL PyOpenGL_accelerate glfw opencv-python")
            print("2. Install a VR runtime:")
            print("   - SteamVR (recommended, supports most headsets)")
            print("   - Oculus Runtime (for Oculus headsets)")
            print("   - Windows Mixed Reality (built into Windows)")
            print("3. Connect your VR headset")
            print(f"{'='*60}\n")
            return ()

        # Check if video file exists
        if not os.path.exists(video_path):
            print(f"\n{'='*60}")
            print("ERROR: Video file not found")
            print(f"{'='*60}")
            print(f"Path: {video_path}")
            print(f"{'='*60}\n")
            return ()

        print(f"\n{'='*60}")
        print("NATIVE VR VIDEO VIEWER")
        print(f"{'='*60}")
        print(f"Video file: {video_path}")
        print(f"Stereo format: {stereo_format}")
        print(f"Swap eyes: {swap_eyes}")
        print(f"Loop: {loop_video}")

        # Map format to internal format strings
        format_map = {
            "Side-by-Side": StereoFormat.SIDE_BY_SIDE,
            "Over-Under": StereoFormat.OVER_UNDER,
            "Mono": StereoFormat.MONO,
        }

        internal_format = format_map.get(stereo_format, StereoFormat.SIDE_BY_SIDE)

        # Map projection type to internal format
        projection_map = {
            "Flat Screen": "flat",
            "Curved Screen": "curved",
            "180° Dome": "dome180",
            "360° Sphere": "sphere360",
        }
        internal_projection = projection_map.get(projection_type, "flat")

        if auto_launch:
            print(f"{'='*60}\n")

            # Map background color to RGB tuple
            bg_color_map = {
                "Black": (0.0, 0.0, 0.0),
                "Dark Gray": (0.2, 0.2, 0.2),
                "Gray": (0.5, 0.5, 0.5),
                "White": (1.0, 1.0, 1.0),
            }
            bg_color = bg_color_map.get(background_color, (0.0, 0.0, 0.0))

            # Launch or update the persistent viewer with video
            success = launch_native_viewer(
                video_path, internal_format, swap_eyes,
                projection_type=internal_projection,
                screen_size=screen_size,
                screen_distance=screen_distance,
                background_color=bg_color,
                is_video=True,
                loop_video=loop_video
            )

            if success:
                print("✓ VR video viewer updated successfully")
                print("✓ Use keyboard controls to play/pause/seek")
            else:
                print("✗ Failed to launch VR video viewer")
        else:
            print("\nAuto-launch disabled.")
            print(f"{'='*60}\n")

        return ()


# Node class mappings for native nodes
NODE_CLASS_MAPPINGS = {
    "NativeStereoImageViewer": NativeStereoImageViewer,
    "NativeStereoVideoViewer": NativeStereoVideoViewer,
    "NativeVRStatus": NativeVRStatus,
}

# Display name mappings
NODE_DISPLAY_NAME_MAPPINGS = {
    "NativeStereoImageViewer": "Native Stereo Image Viewer (PyOpenXR)",
    "NativeStereoVideoViewer": "Native Stereo Video Viewer (PyOpenXR)",
    "NativeVRStatus": "Check Native VR Status",
}
