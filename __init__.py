"""
ComfyStereo - Comprehensive Stereoscopic 3D toolkit for ComfyUI

Combines three powerful stereo tools:
1. Stereo Image Generation - Depth-based stereo conversion with GPU acceleration
2. VR Viewing - Native PyOpenXR viewer for direct VR headset viewing
3. StereoDiffusion - AI-powered stereo generation using diffusion models
"""

# Import base stereo generation nodes (always available)
from .GenerateStereo import NODE_CLASS_MAPPINGS as STEREO_NODES
from .GenerateStereo import NODE_DISPLAY_NAME_MAPPINGS as STEREO_NAMES

# Import native VR viewer nodes (optional, requires PyOpenXR)
try:
    from .native_nodes import NODE_CLASS_MAPPINGS as NATIVE_NODES
    from .native_nodes import NODE_DISPLAY_NAME_MAPPINGS as NATIVE_NAMES
    NATIVE_AVAILABLE = True
except ImportError as e:
    NATIVE_NODES = {}
    NATIVE_NAMES = {}
    NATIVE_AVAILABLE = False
    print("\n" + "="*60)
    print("PyOpenXR not available. Native VR viewer nodes disabled.")
    print("To enable native VR viewing:")
    print("  pip install -r requirements.txt")
    print("="*60 + "\n")

# Import StereoDiffusion nodes (optional, requires diffusers + compatible transformers)
try:
    from .stereodiffusion_nodes import NODE_CLASS_MAPPINGS as DIFFUSION_NODES
    from .stereodiffusion_nodes import NODE_DISPLAY_NAME_MAPPINGS as DIFFUSION_NAMES
    DIFFUSION_AVAILABLE = True
except Exception as e:  # Catch ImportError, RuntimeError from lazy diffusers load, etc.
    DIFFUSION_NODES = {}
    DIFFUSION_NAMES = {}
    DIFFUSION_AVAILABLE = False
    print("\n" + "="*60)
    print("StereoDiffusion (AI-powered stereo) disabled due to import error.")
    print(f"  Reason: {type(e).__name__}: {e}")
    print("This is common with newer transformers versions (>=5.x) + older diffusers.")
    print("Core depth-based StereoImageNode is unaffected and fully functional.")
    print("")
    print("To enable StereoDiffusion (optional):")
    print("  pip install --upgrade 'diffusers>=0.29' 'transformers<5.0' accelerate einops tqdm scikit-image")
    print("  or pin compatible versions in your environment.")
    print("See https://github.com/Dobidop/ComfyStereo/issues for updates.")
    print("="*60 + "\n")

# Combine all node mappings
NODE_CLASS_MAPPINGS = {
    **STEREO_NODES,
    **NATIVE_NODES,
    **DIFFUSION_NODES,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    **STEREO_NAMES,
    **NATIVE_NAMES,
    **DIFFUSION_NAMES,
}

# Print status
print("\n" + "="*60)
print("ComfyStereo - Loaded Modules:")
print("="*60)
print(f"  [OK] Stereo Image Generation ({len(STEREO_NODES)} nodes)")
if NATIVE_AVAILABLE:
    print(f"  [OK] Native VR Viewer ({len(NATIVE_NODES)} nodes)")
else:
    print(f"  [--] Native VR Viewer (not available)")
if DIFFUSION_AVAILABLE:
    print(f"  [OK] StereoDiffusion ({len(DIFFUSION_NODES)} nodes)")
else:
    print(f"  [--] StereoDiffusion (not available)")
print(f"\nTotal: {len(NODE_CLASS_MAPPINGS)} nodes loaded")
print("="*60 + "\n")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']