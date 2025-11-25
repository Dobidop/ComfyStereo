"""
Core VR viewer class with OpenXR rendering
"""

import queue
import time
import gc
import numpy as np

from .constants import (
    PYOPENXR_AVAILABLE, StereoFormat, MediaUpdate,
    STEREO_FORMAT_MAP, PROJECTION_NAMES
)
from .geometry import (
    create_sphere_mesh, create_flat_screen,
    create_curved_screen, create_dome_180
)
from .rendering import (
    create_stereo_shaders, create_help_overlay_shaders,
    setup_vao_vbo, setup_help_overlay_geometry
)
from .media import (
    load_image_texture, calculate_aspect_ratio,
    update_texture_from_frame, VideoCapture
)
from .audio import AudioPlayer
from .controls import (
    create_help_overlay_texture, print_controls_help, KeyboardHandler
)
from .context import GLFWVisibleContextProvider

if PYOPENXR_AVAILABLE:
    import xr
    from xr.utils import Matrix4x4f, GraphicsAPI
    from xr.utils.gl import ContextObject
    from OpenGL import GL
    import glfw
    import cv2


class PersistentNativeViewer:
    """
    Persistent VR viewer that stays running and can receive new images.
    Prevents multiple OpenXR instances and allows continuous updates.
    """

    def __init__(self):
        if not PYOPENXR_AVAILABLE:
            raise ImportError(
                "PyOpenXR is not available. Install with: pip install pyopenxr PyOpenGL pillow"
            )

        # OpenGL resources
        self.texture_id = None
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.sphere_vertices = None
        self.sphere_indices = None

        # Media update queue
        self.media_queue = queue.Queue()
        self.current_media = None
        self.current_format = StereoFormat.SIDE_BY_SIDE
        self.current_swap = False
        self.current_projection = "flat"
        self.current_screen_size = 3.0
        self.current_screen_distance = 3.0
        self.current_aspect_ratio = 16.0 / 9.0  # Default aspect ratio
        self.texture_width = 1920  # Track texture dimensions for aspect ratio recalculation
        self.texture_height = 1080
        self.background_color = (0.0, 0.0, 0.0)  # Background color (RGB 0.0-1.0)

        # Video playback state
        self.is_video = False
        self.video_capture = None
        self.video_playing = True
        self.video_loop = True

        # Audio playback
        self.audio_player = AudioPlayer()

        # Viewer state
        self.running = False
        self.should_stop = False
        self.geometry_needs_update = False
        self.glfw_window = None  # Store GLFW window for keyboard input

        # Alignment offsets for real-time adjustment
        self.horizontal_offset = 0.0
        self.vertical_offset_adjustment = 0.0  # Additional offset beyond auto-centering

        # Help overlay
        self.show_help = True  # Always show help overlay
        self.help_texture_id = None
        self.help_shader_program = None
        self.help_vao = None
        self.help_vbo = None
        self.help_overlay_initialized = False

        # Keyboard handler
        self.keyboard_handler = KeyboardHandler(self)

    def create_geometry(self):
        """Create geometry based on current projection type"""
        if self.current_projection == "flat":
            # Use actual image aspect ratio instead of hardcoded 4:3
            height = self.current_screen_size / self.current_aspect_ratio
            self.sphere_vertices, self.sphere_indices = create_flat_screen(
                width=self.current_screen_size,
                height=height,
                distance=self.current_screen_distance,
                horizontal_offset=self.horizontal_offset,
                vertical_offset=self.vertical_offset_adjustment
            )
        elif self.current_projection == "curved":
            # Use actual image aspect ratio
            height = self.current_screen_size / self.current_aspect_ratio
            self.sphere_vertices, self.sphere_indices = create_curved_screen(
                width=self.current_screen_size,
                height=height,
                distance=self.current_screen_distance,
                curve_amount=0.4,
                horizontal_offset=self.horizontal_offset,
                vertical_offset=self.vertical_offset_adjustment
            )
        elif self.current_projection == "dome180":
            self.sphere_vertices, self.sphere_indices = create_dome_180(
                radius=self.current_screen_distance * 2
            )
        else:  # sphere360
            self.sphere_vertices, self.sphere_indices = create_sphere_mesh(radius=100.0)

    def setup_geometry(self):
        """Set up VAO and VBO for geometry based on projection type"""
        self.create_geometry()
        self.vao, self.vbo, self.ebo = setup_vao_vbo(self.sphere_vertices, self.sphere_indices)

    def render_control_window(self):
        """Render the control window with optional help overlay"""
        if not self.glfw_window:
            return

        # Make the GLFW window's context current
        glfw.make_context_current(self.glfw_window)

        # Set viewport to match window size
        width, height = glfw.get_framebuffer_size(self.glfw_window)
        GL.glViewport(0, 0, width, height)

        # Initialize help overlay resources on first use (must be in GLFW context)
        if not self.help_overlay_initialized:
            try:
                self.help_shader_program = create_help_overlay_shaders()
                self.help_vao, self.help_vbo = setup_help_overlay_geometry()
                self.help_texture_id = create_help_overlay_texture()
                self.help_overlay_initialized = True
                print(f"✓ Help overlay initialized in control window context")
            except Exception as e:
                print(f"⚠️  Failed to initialize help overlay: {e}")
                self.help_overlay_initialized = True  # Don't try again

        # Clear to dark gray background
        GL.glClearColor(0.1, 0.1, 0.1, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        # If help is enabled, render the help overlay
        if self.show_help and self.help_texture_id is not None:
            # Disable depth test for 2D overlay
            GL.glDisable(GL.GL_DEPTH_TEST)

            # Enable blending for transparency
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

            # Use help shader
            GL.glUseProgram(self.help_shader_program)

            # Bind help texture
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.help_texture_id)
            GL.glUniform1i(GL.glGetUniformLocation(self.help_shader_program, "helpTexture"), 0)

            # Draw full-screen quad
            GL.glBindVertexArray(self.help_vao)
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
            GL.glBindVertexArray(0)

            # Restore state
            GL.glDisable(GL.GL_BLEND)
            GL.glEnable(GL.GL_DEPTH_TEST)

        # Swap buffers to display
        glfw.swap_buffers(self.glfw_window)

    def load_media(self, media_path, is_video=False):
        """Load image or video"""
        if is_video:
            self.load_video(media_path)
        else:
            self.load_image(media_path)

    def load_image(self, image_path):
        """Load static image"""
        # Mark as static image (not video)
        self.is_video = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None

        # Stop audio when switching to image
        self.audio_player.stop()

        # Load texture
        self.texture_id, width, height = load_image_texture(image_path, self.texture_id)

        # Store texture dimensions for aspect ratio recalculation
        self.texture_width = width
        self.texture_height = height

        # Calculate and update aspect ratio - only flag update if it actually changed
        aspect_ratio = calculate_aspect_ratio(width, height, self.current_format)
        if abs(self.current_aspect_ratio - aspect_ratio) > 0.01:
            print(f"   Aspect ratio changed: {self.current_aspect_ratio:.3f} → {aspect_ratio:.3f}")
            self.current_aspect_ratio = aspect_ratio
            # Only update geometry for flat/curved projections that depend on aspect ratio
            # And only if we're already running (not during initial load)
            if self.current_projection in ["flat", "curved"] and self.running:
                self.geometry_needs_update = True

    def load_video(self, video_path):
        """Load video file and initialize video capture"""
        try:
            # Close previous video if exists
            if self.video_capture is not None:
                self.video_capture.release()

            # Open video file
            self.video_capture = VideoCapture(video_path)
            self.is_video = True

            # Load audio track
            self.audio_player.load_from_video(video_path, loop=self.video_loop)

            # Load first frame
            ret, frame = self.video_capture.read_frame()
            if ret:
                self.texture_id, width, height = update_texture_from_frame(frame, self.texture_id)

                # Store texture dimensions for aspect ratio recalculation
                self.texture_width = width
                self.texture_height = height

                # Update aspect ratio - only flag update if it actually changed
                aspect_ratio = calculate_aspect_ratio(width, height, self.current_format)
                if abs(self.current_aspect_ratio - aspect_ratio) > 0.01:
                    self.current_aspect_ratio = aspect_ratio
                    # Only update geometry if we're already running (not during initial load)
                    if self.current_projection in ["flat", "curved"] and self.running:
                        self.geometry_needs_update = True

        except Exception as e:
            print(f"   ✗ Error loading video: {e}")
            raise

    def get_next_video_frame(self):
        """Get next frame from video, handling looping"""
        if not self.is_video or self.video_capture is None:
            return False

        ret, frame = self.video_capture.read_frame()

        if ret:
            self.texture_id, _, _ = update_texture_from_frame(frame, self.texture_id)
            return True
        else:
            # End of video
            if self.video_loop:
                # Loop back to start
                self.video_capture.restart()
                ret, frame = self.video_capture.read_frame()
                if ret:
                    self.texture_id, _, _ = update_texture_from_frame(frame, self.texture_id)
                    return True
            return False

    def check_for_updates(self):
        """Check if there's a new media (image or video) to display"""
        try:
            while not self.media_queue.empty():
                update = self.media_queue.get_nowait()

                # Check if this is actually a different media file
                is_same_media = (update.media_path == self.current_media and
                                update.stereo_format == self.current_format and
                                update.swap_eyes == self.current_swap)

                # Check if projection type or size changed (requires geometry rebuild)
                projection_changed = (update.projection_type != self.current_projection or
                                     update.screen_size != self.current_screen_size or
                                     update.screen_distance != self.current_screen_distance)

                if projection_changed:
                    print(f"\n🔄 Projection changed: {self.current_projection} → {update.projection_type}")
                    self.current_projection = update.projection_type
                    self.current_screen_size = update.screen_size
                    self.current_screen_distance = update.screen_distance
                    self.geometry_needs_update = True

                # Update background color
                self.background_color = update.background_color

                # Only reload media if it's actually different
                if not is_same_media or projection_changed:
                    self.current_media = update.media_path
                    self.current_format = update.stereo_format
                    self.current_swap = update.swap_eyes
                    self.video_loop = update.loop_video

                    media_type = "video" if update.is_video else "image"
                    print(f"\n📷 Updating VR view with new {media_type}: {update.media_path}")
                    print(f"   Format: {update.stereo_format}, Swap: {update.swap_eyes}")
                    print(f"   Projection: {update.projection_type}, Size: {update.screen_size}m, Distance: {update.screen_distance}m")

                    # Load video or image
                    self.load_media(self.current_media, update.is_video)
                    if update.is_video:
                        self.video_playing = True
                else:
                    # Same media, just update settings without reloading
                    self.video_loop = update.loop_video
                    print(f"   📝 Media already loaded, skipping reload")

        except queue.Empty:
            pass

    def run(self):
        """Main viewer loop - runs in background thread"""
        # print("🔍 [Core] === Starting run() method ===")
        self.running = True  # Set at start so get_or_create_viewer knows we're running
        self.should_stop = False

        print_controls_help()

        context_provider = None

        try:
            # print("🔍 [Core] Creating GLFWVisibleContextProvider...")
            # Create visible context provider for keyboard input
            context_provider = GLFWVisibleContextProvider()
            # print("🔍 [Core] GLFWVisibleContextProvider created successfully")

            # Ensure GL context is current before creating ContextObject
            # This is critical for xrGetOpenGLGraphicsRequirements to work
            if context_provider._window:
                # print(f"🔍 [Core] Context provider window exists: {context_provider._window}")
                current_before = glfw.get_current_context()
                # print(f"🔍 [Core] Current context before make_current: {current_before}")

                # print("🔍 [Core] Calling glfw.make_context_current...")
                glfw.make_context_current(context_provider._window)

                current_after = glfw.get_current_context()
                # print(f"🔍 [Core] Current context after make_current: {current_after}")
                # print(f"🔍 [Core] Context matches window: {current_after == context_provider._window}")
            else:
                pass
                # print("🔍 [Core] ⚠️ WARNING: Context provider window is None!")

            # print("🔍 [Core] Creating ContextObject (will call xrCreateSession)...")

            # Create instance create info
            instance_info = xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.KHR_OPENGL_ENABLE_EXTENSION_NAME,
                ],
            )
            # print(f"🔍 [Core] InstanceCreateInfo: {instance_info}")

            # Create session create info explicitly
            session_info = xr.SessionCreateInfo()
            # print(f"🔍 [Core] SessionCreateInfo.next before ContextObject: {session_info.next}")

            with ContextObject(
                instance_create_info=instance_info,
                session_create_info=session_info,
                context_provider=context_provider,
            ) as context:
                # print("🔍 [Core] ✓ Inside ContextObject context manager")

                # Get GLFW window and set up keyboard callback
                self.glfw_window = context_provider._window
                glfw.set_key_callback(self.glfw_window, self.keyboard_handler.handle_key)
                print("✓ Keyboard controls enabled (focus the control window to use keys)")

                # Initialize OpenGL resources for VR (in OpenXR context)
                self.shader_program = create_stereo_shaders()
                self.setup_geometry()

                print("⏳ Waiting for media to be loaded...")

                # Enable depth testing
                GL.glEnable(GL.GL_DEPTH_TEST)

                print("✓ VR session started successfully!")
                print("✓ Headset is ready for viewing\n")

                frame_count = 0
                frames_rendered = 0
                last_frame_time = time.time()

                for frame_index, frame_state in enumerate(context.frame_loop()):
                    # Check for stop signal
                    if self.should_stop:
                        print("\n🛑 Stopping VR viewer...")
                        break

                    # Check for media updates
                    # Check immediately on first frame, then every 30 frames
                    if frame_count == 0 or frame_count % 30 == 0:
                        self.check_for_updates()

                        # Rebuild geometry if projection type changed
                        if self.geometry_needs_update:
                            print("🔨 Rebuilding geometry...")
                            self.setup_geometry()
                            self.geometry_needs_update = False
                            print("✓ Geometry updated!")

                    # Poll keyboard events and render control window
                    if self.glfw_window:
                        glfw.poll_events()
                        self.render_control_window()

                    # Advance video frame if playing - sync to audio position
                    if self.is_video and self.video_playing and self.video_capture:
                        # Get audio position for synchronization
                        audio_pos = self.audio_player.get_position()

                        if audio_pos is not None:
                            # Audio-driven sync: calculate which frame we should be on
                            target_frame = int(audio_pos * self.video_capture.fps)
                            current_frame = self.video_capture.current_frame

                            # Add a small buffer to prevent stuttering from rounding errors
                            # Only advance if we're at least 1 full frame behind
                            frames_behind = target_frame - current_frame

                            if frames_behind >= 1:
                                # If we're significantly behind, seek directly
                                if frames_behind > 3:
                                    # Skip directly to target frame
                                    self.video_capture.capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
                                    self.video_capture.current_frame = target_frame
                                    self.get_next_video_frame()
                                else:
                                    # Just read next frame sequentially
                                    self.get_next_video_frame()
                        else:
                            # Fallback to time-based sync if no audio
                            current_time = time.time()
                            elapsed = current_time - last_frame_time

                            if elapsed >= self.video_capture.frame_time:
                                self.get_next_video_frame()
                                last_frame_time = current_time

                    format_int = STEREO_FORMAT_MAP.get(self.current_format, 0)

                    # Render to each eye
                    for view_index, view in enumerate(context.view_loop(frame_state)):
                        # Clear buffers with user-specified background color
                        GL.glClearColor(self.background_color[0], self.background_color[1],
                                       self.background_color[2], 1.0)
                        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

                        if self.texture_id is None:
                            # No image loaded yet, show black background
                            if frame_count == 0:
                                print("⚠️  Waiting for media to load...")
                            continue

                        # Debug: Print first frame render
                        if frames_rendered == 0:
                            print(f"🎬 Rendering first frame (eye {view_index})")

                        frames_rendered += 1

                        # Use shader
                        GL.glUseProgram(self.shader_program)

                        # Set up projection matrix
                        projection = Matrix4x4f.create_projection_fov(
                            graphics_api=GraphicsAPI.OPENGL,
                            fov=view.fov,
                            near_z=0.1,
                            far_z=1000.0,
                        )

                        # Set up view matrix
                        to_view = Matrix4x4f.create_translation_rotation_scale(
                            translation=view.pose.position,
                            rotation=view.pose.orientation,
                            scale=(1, 1, 1),
                        )
                        view_matrix = Matrix4x4f.invert_rigid_body(to_view)

                        # Model matrix
                        model_matrix = np.eye(4, dtype=np.float32)

                        # Set uniforms
                        proj_loc = GL.glGetUniformLocation(self.shader_program, "projection")
                        view_loc = GL.glGetUniformLocation(self.shader_program, "view")
                        model_loc = GL.glGetUniformLocation(self.shader_program, "model")
                        format_loc = GL.glGetUniformLocation(self.shader_program, "stereoFormat")
                        eye_loc = GL.glGetUniformLocation(self.shader_program, "eyeIndex")
                        swap_loc = GL.glGetUniformLocation(self.shader_program, "swapEyes")

                        GL.glUniformMatrix4fv(proj_loc, 1, GL.GL_FALSE, projection.as_numpy().flatten("F"))
                        GL.glUniformMatrix4fv(view_loc, 1, GL.GL_FALSE, view_matrix.as_numpy().flatten("F"))
                        GL.glUniformMatrix4fv(model_loc, 1, GL.GL_FALSE, model_matrix.flatten("F"))
                        GL.glUniform1i(format_loc, format_int)
                        GL.glUniform1i(eye_loc, view_index)
                        GL.glUniform1i(swap_loc, 1 if self.current_swap else 0)

                        # Bind texture
                        GL.glActiveTexture(GL.GL_TEXTURE0)
                        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
                        GL.glUniform1i(GL.glGetUniformLocation(self.shader_program, "texture1"), 0)

                        # Draw geometry
                        GL.glBindVertexArray(self.vao)
                        GL.glDrawElements(
                            GL.GL_TRIANGLES,
                            len(self.sphere_indices),
                            GL.GL_UNSIGNED_INT,
                            None
                        )
                        GL.glBindVertexArray(0)

                    frame_count += 1

                # Cleanup OpenGL resources BEFORE exiting context
                self._cleanup_opengl_resources()

        except KeyboardInterrupt:
            print("\n⚠️  Received interrupt signal (Ctrl+C)")
            print("Note: To stop the viewer, close ComfyUI instead")
        except Exception as e:
            print(f"\n❌ Error in VR viewer: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Final cleanup (context provider and non-OpenGL resources)
            self._cleanup_final(context_provider)

    def _cleanup_opengl_resources(self):
        """Cleanup OpenGL resources while context is still valid"""
        print("\n🧹 Cleaning up OpenGL resources...")

        # Stop audio and video first
        try:
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
            self.audio_player.stop()
        except Exception as e:
            print(f"   ⚠️  Error cleaning up media: {e}")

        # Cleanup OpenGL resources (must be done BEFORE context exits)
        try:
            if self.texture_id:
                GL.glDeleteTextures([self.texture_id])
                self.texture_id = None
            if self.help_texture_id:
                GL.glDeleteTextures([self.help_texture_id])
                self.help_texture_id = None
            if self.vao:
                GL.glDeleteVertexArrays(1, [self.vao])
                self.vao = None
            if self.vbo:
                GL.glDeleteBuffers(1, [self.vbo])
                self.vbo = None
            if self.ebo:
                GL.glDeleteBuffers(1, [self.ebo])
                self.ebo = None
            if self.help_vao:
                GL.glDeleteVertexArrays(1, [self.help_vao])
                self.help_vao = None
            if self.help_vbo:
                GL.glDeleteBuffers(1, [self.help_vbo])
                self.help_vbo = None
            if self.shader_program:
                GL.glDeleteProgram(self.shader_program)
                self.shader_program = None
            if self.help_shader_program:
                GL.glDeleteProgram(self.help_shader_program)
                self.help_shader_program = None
            print("   ✓ OpenGL resources cleaned up")
        except Exception as e:
            print(f"   ⚠️  Error cleaning up OpenGL resources: {e}")

    def _cleanup_final(self, context_provider):
        """Final cleanup after context exits"""
        print("🧹 Final cleanup...")

        # Cleanup context provider and GLFW FIRST
        if context_provider is not None:
            try:
                context_provider.__exit__(None, None, None)
            except Exception as e:
                print(f"   ⚠️  Error cleaning up context provider: {e}")
            context_provider = None

        # Force garbage collection IMMEDIATELY after destroying context
        # This is critical - we need GC to run before we do anything else
        print("   🗑️  Running garbage collection...")
        gc.collect()
        time.sleep(2.0)  # Give GC and OpenXR time to clean up

        # NOW do the rest of the cleanup
        # Clear window reference
        self.glfw_window = None

        # Reset ALL state for clean slate on next run
        self.help_overlay_initialized = False
        self.current_media = None
        self.current_format = StereoFormat.SIDE_BY_SIDE
        self.current_swap = False
        self.current_projection = "flat"
        self.is_video = False
        self.video_playing = True

        # Clear the media queue
        while not self.media_queue.empty():
            try:
                self.media_queue.get_nowait()
            except:
                break

        # Mark as not running
        self.running = False

        print("✓ VR viewer stopped cleanly")
        print("   Clean slate ready for next session")

    def stop(self):
        """Stop the viewer"""
        self.should_stop = True

    def update_media(self, media_path, stereo_format, swap_eyes, projection_type="flat",
                    screen_size=3.0, screen_distance=3.0, background_color=(0.0, 0.0, 0.0),
                    is_video=False, loop_video=True):
        """Queue a new media (image or video) for display"""
        update = MediaUpdate(media_path, stereo_format, swap_eyes, projection_type,
                           screen_size, screen_distance, background_color, is_video, loop_video)
        self.media_queue.put(update)
