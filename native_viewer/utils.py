"""
Utility functions for viewer management and OpenXR checking
"""

import threading
import time
import gc
from .constants import PYOPENXR_AVAILABLE

if PYOPENXR_AVAILABLE:
    import xr

# Global persistent viewer instance
_global_viewer = None
_viewer_thread = None
_viewer_lock = threading.Lock()


def check_openxr_available():
    """
    Check if OpenXR runtime is available

    Returns:
        tuple: (is_available: bool, message: str)
    """
    if not PYOPENXR_AVAILABLE:
        return False, "PyOpenXR not installed"

    try:
        # Try to enumerate available runtimes
        xr.enumerate_instance_extension_properties()
        return True, "OpenXR runtime available"
    except Exception as e:
        return False, f"OpenXR runtime not available: {str(e)}"


def get_or_create_viewer():
    """Get existing viewer or create new one (singleton pattern)"""
    global _global_viewer, _viewer_thread

    # Import here to avoid circular dependency
    from .core import PersistentNativeViewer

    # print("🔍 [Utils] === get_or_create_viewer() called ===")

    with _viewer_lock:
        # If there's a viewer that's still running, return it
        if _global_viewer is not None and _global_viewer.running:
            # print(f"🔍 [Utils] Existing viewer is running, returning it (id: {id(_global_viewer)})")
            return _global_viewer

        # print(f"🔍 [Utils] Global viewer state: {_global_viewer}")
        # print(f"🔍 [Utils] Thread state: {_viewer_thread}")

        # If there's a thread that's still alive, wait for it to finish
        if _viewer_thread is not None and _viewer_thread.is_alive():
            # print(f"🔍 [Utils] Previous thread is alive (id: {_viewer_thread.ident})")
            print("⏳ Waiting for previous viewer instance to terminate...")
            _viewer_thread.join(timeout=10.0)
            if _viewer_thread.is_alive():
                print("⚠️  Previous viewer did not terminate cleanly after 10s")
                return None
            else:
                print("✓ Previous viewer terminated")

            # Give OpenXR a moment to clean up
            print("⏳ Waiting for OpenXR to clean up (3 seconds)...")
            time.sleep(3.0)
            # print("🔍 [Utils] Wait complete")

        # Create new viewer instance
        print("🔨 Creating new VR viewer instance...")
        _global_viewer = PersistentNativeViewer()
        # print(f"🔍 [Utils] New viewer created (id: {id(_global_viewer)})")

        # print(f"🔍 [Utils] Starting viewer thread...")
        _viewer_thread = threading.Thread(target=_global_viewer.run, daemon=True)
        _viewer_thread.start()
        # print(f"🔍 [Utils] Thread started (id: {_viewer_thread.ident})")

        # Give it a moment to initialize
        # print("🔍 [Utils] Sleeping 1.5s for initialization...")
        time.sleep(1.5)
        # print(f"🔍 [Utils] Returning viewer (running={_global_viewer.running})")

        return _global_viewer


def stop_global_viewer():
    """Stop the global viewer"""
    global _global_viewer
    if _global_viewer:
        _global_viewer.stop()


def launch_native_viewer(media_path, stereo_format="sbs", swap_eyes=False,
                        projection_type="flat", screen_size=3.0, screen_distance=3.0,
                        background_color=(0.0, 0.0, 0.0), is_video=False, loop_video=True):
    """
    Launch or update the native viewer with a new image or video.
    If viewer is already running, updates it with the new media.
    Otherwise, starts a new viewer.

    Args:
        media_path: Path to stereo image or video
        stereo_format: Stereo format (sbs, ou, mono)
        swap_eyes: Whether to swap eyes
        projection_type: Projection type (flat, curved, dome180, sphere360)
        screen_size: Screen size in meters (for flat/curved/dome)
        screen_distance: Distance from viewer in meters
        background_color: Background color as RGB tuple (0.0-1.0 range)
        is_video: Whether the media is a video file
        loop_video: Whether to loop video playback

    Returns:
        bool: True if successful, False if error
    """
    available, message = check_openxr_available()

    if not available:
        print(f"ERROR: {message}")
        print("\nTo use native VR viewer:")
        print("1. Install PyOpenXR: pip install pyopenxr PyOpenGL pillow opencv-python")
        print("2. Install SteamVR or Oculus runtime")
        print("3. Make sure your VR headset is connected")
        return False

    try:
        viewer = get_or_create_viewer()

        if viewer is None:
            print("❌ Failed to create or get viewer instance")
            print("💡 The previous viewer may still be cleaning up.")
            print("   Please wait a few seconds and try again.")
            return False

        # Queue the media update - it will be processed by the render loop
        viewer.update_media(media_path, stereo_format, swap_eyes, projection_type,
                          screen_size, screen_distance, background_color, is_video, loop_video)

        return True

    except Exception as e:
        print(f"Error launching native viewer: {e}")
        import traceback
        traceback.print_exc()
        return False
