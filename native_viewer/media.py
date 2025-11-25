"""
Media loading and texture management for images and videos
"""

import time
import numpy as np
from .constants import PYOPENXR_AVAILABLE, StereoFormat

if PYOPENXR_AVAILABLE:
    from PIL import Image
    from OpenGL import GL
    import cv2


def load_image_texture(image_path, texture_id=None):
    """
    Load image as OpenGL texture

    Returns:
        tuple: (texture_id, width, height, aspect_ratio)
    """
    if not PYOPENXR_AVAILABLE:
        raise ImportError("PyOpenXR dependencies not available")

    print(f"   Loading texture from: {image_path}")

    img = Image.open(image_path)
    img = img.convert('RGB')
    img_data = np.array(img, dtype=np.uint8)

    print(f"   Image size: {img.width}x{img.height}, channels: {img_data.shape}")

    if texture_id is None:
        # Create new texture
        texture_id = GL.glGenTextures(1)
        print(f"   Created texture ID: {texture_id}")

    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)

    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    # Use GL_SRGB8 internal format to handle sRGB color space correctly
    # This prevents the washed-out look caused by treating sRGB data as linear
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_SRGB8,
        img.width, img.height, 0,
        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, img_data
    )

    print(f"   ✓ Texture loaded successfully!")

    return texture_id, img.width, img.height


def calculate_aspect_ratio(width, height, stereo_format):
    """
    Calculate aspect ratio based on image dimensions and stereo format

    Args:
        width: Image width in pixels
        height: Image height in pixels
        stereo_format: Stereo format (sbs, ou, mono, etc.)

    Returns:
        float: Aspect ratio (per-eye for stereo formats)
    """
    if stereo_format == StereoFormat.SIDE_BY_SIDE:
        # For side-by-side, divide width by 2 to get per-eye aspect ratio
        aspect_ratio = (width / 2.0) / height
    elif stereo_format == StereoFormat.OVER_UNDER:
        # For over-under, divide height by 2
        aspect_ratio = width / (height / 2.0)
    else:
        # Mono or anaglyph use full image
        aspect_ratio = width / height

    return aspect_ratio


def update_texture_from_frame(frame, texture_id=None):
    """
    Update OpenGL texture with a video frame

    Args:
        frame: OpenCV frame (BGR format)
        texture_id: Existing texture ID or None to create new

    Returns:
        tuple: (texture_id, width, height)
    """
    if not PYOPENXR_AVAILABLE:
        raise ImportError("PyOpenXR dependencies not available")

    # OpenCV uses BGR, convert to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    height, width = frame_rgb.shape[:2]

    if texture_id is None:
        texture_id = GL.glGenTextures(1)

    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    # Use GL_SRGB8 internal format to handle sRGB color space correctly
    # This prevents the washed-out look caused by treating sRGB data as linear
    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_SRGB8,
        width, height, 0,
        GL.GL_RGB, GL.GL_UNSIGNED_BYTE, frame_rgb
    )

    return texture_id, width, height


class VideoCapture:
    """Wrapper for OpenCV video capture with additional functionality"""

    def __init__(self, video_path):
        if not PYOPENXR_AVAILABLE:
            raise ImportError("PyOpenXR dependencies not available")

        print(f"   Loading video from: {video_path}")

        self.capture = cv2.VideoCapture(video_path)

        if not self.capture.isOpened():
            raise RuntimeError(f"Could not open video file: {video_path}")

        # Get video properties
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if self.fps <= 0:
            self.fps = 30.0  # Default to 30fps if unknown

        self.frame_time = 1.0 / self.fps
        self.current_frame = 0
        self.last_frame_time = time.time()

        print(f"   Video loaded: {self.width}x{self.height}, {self.fps} fps, {self.total_frames} frames")

    def read_frame(self):
        """
        Read next frame from video

        Returns:
            tuple: (success, frame) or (False, None) if no frame available
        """
        ret, frame = self.capture.read()
        if ret:
            self.current_frame += 1
        return ret, frame

    def seek(self, frame_offset):
        """Seek video by number of frames (positive or negative)"""
        new_frame = max(0, min(self.current_frame + frame_offset, self.total_frames - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
        self.current_frame = new_frame
        print(f"   Seeked to frame {self.current_frame}/{self.total_frames}")

    def restart(self):
        """Restart video from beginning"""
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.current_frame = 0
        print("   Video restarted")

    def release(self):
        """Release video capture resources"""
        if self.capture is not None:
            self.capture.release()
            self.capture = None
