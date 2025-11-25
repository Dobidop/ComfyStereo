"""
Constants and data classes for the native VR viewer
"""

import sys

# Check for PyOpenXR availability
try:
    import xr
    from xr.utils import Matrix4x4f, GraphicsAPI
    from xr.utils.gl import ContextObject
    from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider
    from OpenGL import GL
    from PIL import Image
    import cv2
    import glfw
    import pygame
    import subprocess
    import tempfile
    PYOPENXR_AVAILABLE = True
except ImportError as e:
    PYOPENXR_AVAILABLE = False
    print(f"PyOpenXR not available. Install with: pip install pyopenxr PyOpenGL pillow opencv-python pygame")
    print(f"Error: {e}")


class StereoFormat:
    """Stereo format constants"""
    SIDE_BY_SIDE = "sbs"
    OVER_UNDER = "ou"
    ANAGLYPH = "anaglyph"
    MONO = "mono"
    SEPARATE = "separate"


class MediaUpdate:
    """Represents a media (image or video) update for the viewer"""
    def __init__(self, media_path, stereo_format, swap_eyes, projection_type="flat",
                 screen_size=3.0, screen_distance=3.0, background_color=(0.0, 0.0, 0.0),
                 is_video=False, loop_video=True):
        self.media_path = media_path
        self.stereo_format = stereo_format
        self.swap_eyes = swap_eyes
        self.projection_type = projection_type
        self.screen_size = screen_size
        self.screen_distance = screen_distance
        self.background_color = background_color
        self.is_video = is_video
        self.loop_video = loop_video


# Stereo format mapping for shader (format string -> integer)
STEREO_FORMAT_MAP = {
    StereoFormat.SIDE_BY_SIDE: 0,
    StereoFormat.OVER_UNDER: 1,
    StereoFormat.ANAGLYPH: 2,
    StereoFormat.MONO: 2,
}

# Projection type mappings
PROJECTION_NAMES = {
    "flat": "Flat Screen",
    "curved": "Curved Screen",
    "dome180": "180° Dome",
    "sphere360": "360° Sphere"
}

# Stereo format names
FORMAT_NAMES = {
    "sbs": "Side-by-Side",
    "ou": "Over-Under",
    "mono": "Mono"
}
