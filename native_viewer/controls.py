"""
Keyboard controls and help overlay for VR viewer
"""

import numpy as np
from .constants import PYOPENXR_AVAILABLE, FORMAT_NAMES

if PYOPENXR_AVAILABLE:
    from PIL import Image, ImageDraw, ImageFont
    from OpenGL import GL
    import glfw


def create_help_overlay_texture(texture_id=None):
    """Create a texture with help text to display in the control window"""
    if not PYOPENXR_AVAILABLE:
        return None

    # Create image for help text
    width, height = 400, 300
    img = Image.new('RGBA', (width, height), (20, 20, 20, 230))  # Semi-transparent dark background
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fallback to default
    try:
        # Platform-agnostic font loading
        import platform
        system = platform.system()
        if system == "Windows":
            font = ImageFont.truetype("arial.ttf", 11)
            font_bold = ImageFont.truetype("arialbd.ttf", 12)
        elif system == "Darwin":  # macOS
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
            font_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
        else:  # Linux
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
            font_bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        font_bold = font

    # Draw help text
    y = 10
    line_height = 14

    # Title
    draw.text((10, y), "KEYBOARD CONTROLS", fill=(255, 255, 100), font=font_bold)
    y += line_height + 5

    draw.text((10, y), "(focus this windows to use controls)", fill=(255, 255, 100), font=font_bold)
    y += line_height + 5

    # Video playback
    draw.text((10, y), "VIDEO:", fill=(100, 200, 255), font=font_bold)
    y += line_height
    draw.text((10, y), "  SPACE - Play/Pause", fill=(220, 220, 220), font=font)
    y += line_height
    draw.text((10, y), "  R - Restart    L - Toggle loop", fill=(220, 220, 220), font=font)
    y += line_height
    draw.text((10, y), "  Arrows - Seek (1s/5s)", fill=(220, 220, 220), font=font)
    y += line_height + 3

    # Viewer adjustments
    draw.text((10, y), "VIEWER:", fill=(100, 200, 255), font=font_bold)
    y += line_height
    draw.text((10, y), "  P - Cycle projection", fill=(220, 220, 220), font=font)
    y += line_height
    draw.text((10, y), "  PgUp/PgDn - Screen distance", fill=(220, 220, 220), font=font)
    y += line_height
    draw.text((10, y), "  +/- - Screen size", fill=(220, 220, 220), font=font)
    y += line_height
    draw.text((10, y), "  Shift+S - Stereo format", fill=(220, 220, 220), font=font)
    y += line_height
    draw.text((10, y), "  E - Swap eyes", fill=(220, 220, 220), font=font)
    y += line_height + 3

    # Alignment
    draw.text((10, y), "ALIGN:", fill=(100, 200, 255), font=font_bold)
    y += line_height
    draw.text((10, y), "  W/A/S/D - Move screen", fill=(220, 220, 220), font=font)
    y += line_height
    draw.text((10, y), "  0 - Reset to center", fill=(220, 220, 220), font=font)
    y += line_height + 3

    # Other
    draw.text((10, y), "OTHER:", fill=(100, 200, 255), font=font_bold)
    y += line_height
    draw.text((10, y), "  Q/ESC - Quit viewer", fill=(220, 220, 220), font=font)

    # Convert to OpenGL texture
    img_data = np.array(img)

    if texture_id is None:
        texture_id = GL.glGenTextures(1)

    GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)

    GL.glTexImage2D(
        GL.GL_TEXTURE_2D, 0, GL.GL_RGBA,
        width, height, 0,
        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, img_data
    )

    return texture_id


def print_controls_help():
    """Print keyboard controls to console"""
    print("\n" + "="*60)
    print("🥽 NATIVE VR VIEWER STARTING")
    print("="*60)
    print("PUT ON YOUR HEADSET NOW!")
    print("\n📖 KEYBOARD CONTROLS (focus control window):")
    print("="*60)
    print("\n🎬 VIDEO PLAYBACK:")
    print("  SPACE      - Play/Pause")
    print("  R          - Restart video")
    print("  LEFT/RIGHT - Seek backward/forward 1 second")
    print("  DOWN/UP    - Seek backward/forward 5 seconds")
    print("  L          - Toggle loop")
    print("\n📐 VIEWER ADJUSTMENTS:")
    print("  P          - Cycle projection type")
    print("  PgUp/PgDn  - Increase/Decrease screen distance")
    print("  + / -      - Increase/Decrease screen size")
    print("  Shift+S    - Cycle stereo format")
    print("  E          - Toggle swap eyes")
    print("\n🎯 ALIGNMENT:")
    print("  W / S      - Move screen up/down")
    print("  A / D      - Move screen left/right")
    print("  0          - Reset alignment to center")
    print("\n🚪 OTHER:")
    print("  Q or ESC   - Quit viewer (ComfyUI keeps running)")
    print("="*60 + "\n")


class KeyboardHandler:
    """Handles keyboard input for viewer control"""

    def __init__(self, viewer):
        """
        Initialize keyboard handler

        Args:
            viewer: Reference to the PersistentNativeViewer instance
        """
        self.viewer = viewer

    def handle_key(self, window, key, scancode, action, mods):
        """
        Handle keyboard input for video controls and viewer adjustments

        Args:
            window: GLFW window
            key: Key code
            scancode: Platform-specific scancode
            action: GLFW_PRESS, GLFW_RELEASE, or GLFW_REPEAT
            mods: Modifier keys (Shift, Ctrl, Alt, etc.)
        """
        if not PYOPENXR_AVAILABLE:
            return

        if action != glfw.PRESS and action != glfw.REPEAT:
            return

        # Video playback controls
        if key == glfw.KEY_SPACE:
            self._handle_play_pause()
        elif key == glfw.KEY_R:
            self._handle_restart()
        elif key == glfw.KEY_LEFT and not (mods & glfw.MOD_SHIFT):
            self._handle_seek(-1)
        elif key == glfw.KEY_RIGHT and not (mods & glfw.MOD_SHIFT):
            self._handle_seek(1)
        elif key == glfw.KEY_DOWN and not (mods & glfw.MOD_SHIFT):
            self._handle_seek(-5)
        elif key == glfw.KEY_UP and not (mods & glfw.MOD_SHIFT):
            self._handle_seek(5)
        elif key == glfw.KEY_L:
            self._handle_loop_toggle()

        # Viewer controls
        elif key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
            self._handle_quit()
        elif key == glfw.KEY_P:
            self._handle_projection_cycle()
        elif key == glfw.KEY_PAGE_UP:
            self._handle_distance_adjust(0.5)
        elif key == glfw.KEY_PAGE_DOWN:
            self._handle_distance_adjust(-0.5)
        elif key == glfw.KEY_MINUS or key == glfw.KEY_KP_SUBTRACT:
            self._handle_size_adjust(-0.5)
        elif key == glfw.KEY_EQUAL or key == glfw.KEY_KP_ADD:
            self._handle_size_adjust(0.5)
        elif key == glfw.KEY_S and (mods & glfw.MOD_SHIFT):
            self._handle_format_cycle()
        elif key == glfw.KEY_E:
            self._handle_swap_eyes()

        # Alignment controls
        elif key == glfw.KEY_W:
            self._handle_vertical_move(0.1)
        elif key == glfw.KEY_S and not (mods & glfw.MOD_SHIFT):
            self._handle_vertical_move(-0.1)
        elif key == glfw.KEY_A:
            self._handle_horizontal_move(-0.1)
        elif key == glfw.KEY_D:
            self._handle_horizontal_move(0.1)
        elif key == glfw.KEY_0:
            self._handle_reset_alignment()
        # H key removed - help overlay is now always visible

    # Video control handlers
    def _handle_play_pause(self):
        """Toggle video playback"""
        if self.viewer.is_video:
            self.viewer.video_playing = not self.viewer.video_playing
            self.viewer.audio_player.toggle()
            status = "Playing" if self.viewer.video_playing else "Paused"
            print(f"   Video {status}")

    def _handle_restart(self):
        """Restart video"""
        if self.viewer.is_video and self.viewer.video_capture:
            self.viewer.video_capture.restart()
            self.viewer.audio_player.restart()

    def _handle_seek(self, seconds):
        """Seek video and audio by seconds"""
        if self.viewer.is_video and self.viewer.video_capture:
            # Seek video
            frames = int(self.viewer.video_capture.fps * seconds)
            self.viewer.video_capture.seek(frames)
            # Seek audio to match video position
            video_time = self.viewer.video_capture.current_frame / self.viewer.video_capture.fps
            self.viewer.audio_player.seek(video_time)

    def _handle_loop_toggle(self):
        """Toggle video looping"""
        self.viewer.video_loop = not self.viewer.video_loop
        status = "enabled" if self.viewer.video_loop else "disabled"
        print(f"   Video loop {status}")

    # Viewer control handlers
    def _handle_quit(self):
        """Quit viewer"""
        print("\n👋 Quitting VR viewer (ComfyUI continues running)...")
        self.viewer.should_stop = True

    def _handle_projection_cycle(self):
        """Cycle through projection types"""
        projections = ["flat", "curved", "dome180", "sphere360"]
        from .constants import PROJECTION_NAMES

        current_idx = projections.index(self.viewer.current_projection) if self.viewer.current_projection in projections else 0
        next_idx = (current_idx + 1) % len(projections)
        self.viewer.current_projection = projections[next_idx]
        self.viewer.geometry_needs_update = True
        print(f"   📐 Projection: {PROJECTION_NAMES.get(self.viewer.current_projection, self.viewer.current_projection)}")

    def _handle_distance_adjust(self, delta):
        """Adjust screen distance"""
        self.viewer.current_screen_distance = max(1.0, min(10.0, self.viewer.current_screen_distance + delta))
        self.viewer.geometry_needs_update = True
        print(f"   📏 Screen distance: {self.viewer.current_screen_distance:.1f}m")

    def _handle_size_adjust(self, delta):
        """Adjust screen size"""
        self.viewer.current_screen_size = max(1.0, min(10.0, self.viewer.current_screen_size + delta))
        self.viewer.geometry_needs_update = True
        print(f"   📺 Screen size: {self.viewer.current_screen_size:.1f}m")

    def _handle_format_cycle(self):
        """Cycle through stereo formats"""
        from .media import calculate_aspect_ratio

        formats = ["sbs", "ou", "mono"]
        current_idx = formats.index(self.viewer.current_format) if self.viewer.current_format in formats else 0
        next_idx = (current_idx + 1) % len(formats)
        self.viewer.current_format = formats[next_idx]

        # Recalculate aspect ratio based on new format and current texture dimensions
        new_aspect_ratio = calculate_aspect_ratio(
            self.viewer.texture_width,
            self.viewer.texture_height,
            self.viewer.current_format
        )
        if abs(self.viewer.current_aspect_ratio - new_aspect_ratio) > 0.01:
            self.viewer.current_aspect_ratio = new_aspect_ratio

        self.viewer.geometry_needs_update = True
        print(f"   🎬 Stereo format: {FORMAT_NAMES.get(self.viewer.current_format, self.viewer.current_format)}")

    def _handle_swap_eyes(self):
        """Toggle eye swapping"""
        self.viewer.current_swap = not self.viewer.current_swap
        status = "ON" if self.viewer.current_swap else "OFF"
        print(f"   👁️  Swap eyes: {status}")

    # Alignment handlers
    def _handle_vertical_move(self, delta):
        """Move screen vertically"""
        self.viewer.vertical_offset_adjustment += delta
        self.viewer.geometry_needs_update = True
        arrow = "⬆️" if delta > 0 else "⬇️"
        print(f"   {arrow}  Vertical offset: {self.viewer.vertical_offset_adjustment:+.2f}m")

    def _handle_horizontal_move(self, delta):
        """Move screen horizontally"""
        self.viewer.horizontal_offset += delta
        self.viewer.geometry_needs_update = True
        arrow = "➡️" if delta > 0 else "⬅️"
        print(f"   {arrow}  Horizontal offset: {self.viewer.horizontal_offset:+.2f}m")

    def _handle_reset_alignment(self):
        """Reset alignment offsets"""
        self.viewer.vertical_offset_adjustment = 0.0
        self.viewer.horizontal_offset = 0.0
        self.viewer.geometry_needs_update = True
        print(f"   🔄 Reset alignment offsets to center")

    def _handle_help_toggle(self):
        """Toggle help overlay"""
        self.viewer.show_help = not self.viewer.show_help
        status = "ON" if self.viewer.show_help else "OFF"
        print(f"   ❓ Help overlay: {status}")
