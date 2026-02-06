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

# Import StereoDiffusion nodes (optional, requires diffusers)
try:
    from .stereodiffusion_nodes import NODE_CLASS_MAPPINGS as DIFFUSION_NODES
    from .stereodiffusion_nodes import NODE_DISPLAY_NAME_MAPPINGS as DIFFUSION_NAMES
    DIFFUSION_AVAILABLE = True
except ImportError as e:
    DIFFUSION_NODES = {}
    DIFFUSION_NAMES = {}
    DIFFUSION_AVAILABLE = False
    print("\n" + "="*60)
    print("StereoDiffusion dependencies not available. AI stereo generation disabled.")
    print("To enable StereoDiffusion:")
    print("  pip install diffusers transformers accelerate einops tqdm scikit-image")
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