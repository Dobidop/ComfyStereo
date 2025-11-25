"""
GLFW context provider for OpenXR with visible window
"""

from .constants import PYOPENXR_AVAILABLE

if PYOPENXR_AVAILABLE:
    from xr.utils.gl.glfw_util import GLFWOffscreenContextProvider
    import glfw

    class GLFWVisibleContextProvider(GLFWOffscreenContextProvider):
        """
        GLFW context provider that creates a VISIBLE window for keyboard input.
        Extends the offscreen provider but overrides window creation.
        """
        def __init__(self):
            # print("🔍 [Context] Starting GLFWVisibleContextProvider.__init__")

            # Don't call parent __init__ - we'll do our own initialization
            # Initialize GLFW if not already done
            # print("🔍 [Context] Calling glfw.init()...")
            init_result = glfw.init()
            # print(f"🔍 [Context] glfw.init() returned: {init_result}")
            if not init_result:
                raise RuntimeError("Failed to initialize GLFW")

            # Set window hints for visible window
            # print("🔍 [Context] Setting window hints...")
            glfw.window_hint(glfw.VISIBLE, True)  # Make window VISIBLE (different from parent)
            glfw.window_hint(glfw.DOUBLEBUFFER, False)
            glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
            glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
            glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
            glfw.window_hint(glfw.RESIZABLE, False)
            glfw.window_hint(glfw.FLOATING, True)  # Keep on top

            # Create a small visible window for controls
            # print("🔍 [Context] Creating GLFW window...")
            self._window = glfw.create_window(400, 300, "VR Video Controls", None, None)
            # print(f"🔍 [Context] Window created: {self._window}")
            if not self._window:
                glfw.terminate()
                raise RuntimeError("Failed to create visible GLFW window")

            # Make context current
            # print("🔍 [Context] Making context current...")
            glfw.make_context_current(self._window)
            current_context = glfw.get_current_context()
            # print(f"🔍 [Context] Current context after make_current: {current_context}")
            # print(f"🔍 [Context] Context matches window: {current_context == self._window}")
            glfw.swap_interval(0)
            # print("🔍 [Context] GLFWVisibleContextProvider.__init__ complete")

        def make_current(self) -> None:
            """Activate this OpenGL context for subsequent GL calls."""
            # print(f"🔍 [Context] make_current() called for window: {self._window}")
            if not self._window:
                # print("🔍 [Context] ⚠️ WARNING: Window is None in make_current()!")
                raise RuntimeError("Cannot make context current - window is None")
            glfw.make_context_current(self._window)
            current = glfw.get_current_context()
            # print(f"🔍 [Context] Context after make_current: {current}")

        def done_current(self) -> None:
            """Unbind this context from the current thread."""
            # print("🔍 [Context] done_current() called")
            glfw.make_context_current(None)

        def destroy(self) -> None:
            """Tear down the window (called by OpenGLGraphics.destroy())"""
            # print("🔍 [Context] destroy() called")
            if self._window:
                glfw.destroy_window(self._window)
                self._window = None
            # DON'T terminate GLFW here - let __exit__ do that

        def __enter__(self):
            # print("🔍 [Context] __enter__ called")
            current_context = glfw.get_current_context()
            # print(f"🔍 [Context] Current context in __enter__: {current_context}")
            # print(f"🔍 [Context] Context matches window: {current_context == self._window}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            # print("🔍 [Context] __exit__ called")
            if self._window:
                # print(f"🔍 [Context] Window exists: {self._window}")
                current_context = glfw.get_current_context()
                # print(f"🔍 [Context] Current context before unbind: {current_context}")

                # Unbind context before destroying window
                # print("🔍 [Context] Unbinding context...")
                glfw.make_context_current(None)
                current_context = glfw.get_current_context()
                # print(f"🔍 [Context] Current context after unbind: {current_context}")

                # print("🔍 [Context] Destroying window...")
                glfw.destroy_window(self._window)
                self._window = None
                # print("🔍 [Context] Window destroyed")

            # Terminate GLFW to ensure clean state for next run
            # print("🔍 [Context] Terminating GLFW...")
            glfw.terminate()
            # print("🔍 [Context] GLFW terminated - clean slate for next run")
else:
    # Stub class when PyOpenXR is not available
    GLFWVisibleContextProvider = None
