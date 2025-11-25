# ComfyStereo - Complete Stereoscopic 3D Toolkit for ComfyUI

A comprehensive stereoscopic 3D toolkit for ComfyUI that combines three powerful solutions into one unified package:

1. **Stereo Image Generation** - Depth-based stereo conversion with GPU acceleration
2. **Native VR Viewing** - PyOpenXR viewer for direct VR headset viewing
3. **StereoDiffusion** - AI-powered stereo generation using diffusion models

## Features Overview

### Core Stereo Generation
- **GPU-Accelerated Processing** - 5-20x faster depth processing with CUDA, but lower quality
- **Advanced Fill Techniques** - Multiple interpolation methods (Polylines, Naive, Hybrid Edge, GPU Warp)
- **Edge-Aware Depth Blurring** - Reduces artifacts at high divergence settings
- **Multiple Output Formats** - Side-by-Side, Top-Bottom, Red-Cyan Anaglyph
- **Batch Video Processing** - Memory-efficient video frame processing

### Native VR Viewing
- **Auto-Launch to Headset** - Direct VR viewing without browser
- **Multiple Stereo Formats** - Side-by-Side, Over-Under, Mono
- **Projection Options** - Flat, Curved, 180° Dome, 360° Sphere
- **Image & Video Support** - View both stereo images and videos
- **All VR Headsets** - Quest, Vive, Index, WMR, and more

### StereoDiffusion AI
- **AI-Powered Generation** - Uses diffusion models for stereo creation
- **DDIM Inversion** - Null-text optimization for high-quality reconstruction
- **Bilateral Neighbor Attention** - Stereo-consistent diffusion
- **ComfyUI Native Models** - Works with MODEL/CLIP/VAE inputs
- **Diffusers Support** - Also works with HuggingFace model IDs

## Installation

### Method 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "ComfyStereo"
3. Click Install
4. Restart ComfyUI

### Method 2: Manual Installation

1. Clone the repository:
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/Dobidop/ComfyStereo.git
```

2. Install core dependencies:
```bash
cd ComfyStereo
pip install -r requirements.txt
```

3. Restart ComfyUI

## Available Nodes

### Stereo Image Generation Nodes

#### 1. Stereo Image Node
The main node for depth-based stereo conversion.

**Inputs:**
- `image` (IMAGE) - Source image
- `depth_map` (IMAGE) - Depth map (grayscale)
- `divergence` (FLOAT) - Stereo effect strength (0.05-15.0, default: 3.5)
- `separation` (FLOAT) - Additional horizontal shift (-5.0 to 5.0)
- `stereo_balance` (FLOAT) - Effect distribution between eyes (-0.95 to 0.95)
- `convergence_point` (FLOAT) - Depth level at screen plane (0.0-1.0, default: 0.5)
- `modes` - Output format: left-right, right-left, top-bottom, bottom-top, red-cyan-anaglyph
- `fill_technique` - Infill method (see [Infill Methods](#infill-methods))
- `depth_map_blur` (BOOLEAN) - Enable edge-aware depth blurring
- `depth_blur_edge_threshold` (FLOAT) - Gradient sharpness cutoff (0.1-15.0)
- `batch_size` (INT) - Frames per memory cleanup cycle

**Outputs:**
- `stereoscope` (IMAGE) - Final stereo image
- `blurred_depthmap_left` (IMAGE) - Processed left depth map
- `blurred_depthmap_right` (IMAGE) - Processed right depth map
- `no_fill_imperfect_mask` (MASK) - Unfilled region mask

#### 2. DeoVR View Node
Launch and view stereo content in DeoVR.

**Inputs:**
- `file_name` (STRING) - File or video path
- `projection_type` - Display mode (Flat, 180°, 360°, Fisheye variants)
- `eye_location` - Side-by-Side or Top-Bottom
- `file_location` - Output folder, Input folder, or Other

**Setup Required:**
Edit `config.json` to set your DeoVR.exe path. Enable "Remote control" in DeoVR developer settings.

### Native VR Viewer Nodes

Requires `pip install -r requirements-native.txt` and a VR runtime (SteamVR, Oculus, or WMR).

#### 3. Native Stereo Image Viewer
Auto-launches images directly into VR headset.

**Inputs:**
- `image` (IMAGE) - Stereo image
- `stereo_format` - Side-by-Side, Over-Under, Mono
- `projection_type` - Flat Screen, Curved Screen, 180° Dome, 360° Sphere
- `screen_size` (FLOAT) - Virtual screen size (1.0-10.0)
- `screen_distance` (FLOAT) - Distance from viewer (1.0-10.0)
- `swap_eyes` (BOOLEAN) - Swap left/right
- `auto_launch` (BOOLEAN) - Launch into headset
- `background_color` - Black, Dark Gray, Gray, White
- `right_image` (IMAGE, optional) - Separate right eye image

**Outputs:**
- `passthrough` (IMAGE) - Original image

#### 4. Native Stereo Video Viewer
Play stereo videos in VR with keyboard controls.

**Inputs:**
- `video_path` (STRING) - Path to stereo video
- `stereo_format` - Side-by-Side, Over-Under, Mono
- `projection_type` - Flat Screen, Curved Screen, 180° Dome, 360° Sphere
- `screen_size` (FLOAT) - Virtual screen size
- `screen_distance` (FLOAT) - Distance from viewer
- `swap_eyes` (BOOLEAN)
- `loop_video` (BOOLEAN)
- `auto_launch` (BOOLEAN)
- `background_color`

#### 5. Native VR Status
Check PyOpenXR installation and VR runtime availability.

**Outputs:**
- `status_message` (STRING) - Diagnostic information
- `is_available` (BOOLEAN) - VR readiness

### StereoDiffusion AI Nodes

Requires `pip install -r requirements-diffusion.txt` and 8GB+ VRAM.

#### 6. StereoDiffusion Node
AI-powered stereo generation using diffusion models.

**Inputs:**
- `image` (IMAGE) - Source image
- `depth_map` (IMAGE) - Depth map
- `scale_factor` (FLOAT) - Disparity strength (1.0-20.0, default: 9.0)
- `direction` - "uni" (unidirectional) or "bi" (bidirectional) attention
- `deblur` (BOOLEAN) - Add noise to unfilled regions
- `num_ddim_steps` (INT) - DDIM steps (10-100, default: 50)
- `null_text_optimization` (BOOLEAN) - Enable for better quality (slower)
- `guidance_scale` (FLOAT) - CFG scale (1.0-20.0, default: 7.5)
- `model` (MODEL, optional) - ComfyUI MODEL from checkpoint loader
- `clip` (CLIP, optional) - ComfyUI CLIP
- `vae` (VAE, optional) - ComfyUI VAE
- `model_id` (STRING) - HuggingFace model ID (fallback if MODEL/CLIP/VAE not provided)

**Outputs:**
- `stereo_pair` (IMAGE) - Side-by-side stereo image
- `left_image` (IMAGE) - Left eye view
- `right_image` (IMAGE) - Right eye view

**Supports:** SD1.x and SD2.x models (SDXL/FLUX planned)

## Key Parameters Explained

### Divergence
Controls the **strength of the 3D effect**. Higher values = more depth perception.
- Low (1-3): Subtle depth
- Medium (3-7): Balanced effect
- High (7-15): Extreme pop-out

### Convergence Point
Controls which depth appears **at screen plane** (zero parallax).
- `0.0` = Nearest depth at screen → Content recedes behind screen
- `0.5` = Mid-depth at screen → Balanced (default)
- `1.0` = Furthest depth at screen → Content pops toward viewer

**Use cases:**
- **Pop-out mode** (1.0): Product displays, comics
- **Window mode** (0.0): Subtle depth, natural recession
- **Portrait mode** (0.6-0.7): Face at screen, background recedes
- **Landscape mode** (0.3-0.4): Foreground pops, horizon recedes

### Stereo Balance
Distributes divergence between eyes.
- `0.0` = Even distribution
- Positive/negative = Shift effect toward one eye

### Separation
Additional horizontal shift percentage (independent of depth).

## Infill Methods

| Method | Speed | Quality | Description |
|--------|-------|---------|-------------|
| **GPU Warp (Fast)** | ⚡⚡⚡ | ⭐⭐⭐ | GPU-accelerated warping with border padding |
| **Polylines Soft** | ⚡⚡ | ⭐⭐⭐⭐ | Best general filler, maintains structure |
| **Polylines Sharp** | ⚡⚡ | ⭐⭐⭐⭐ | Similar to Soft but sharper transitions |
| **Hybrid Edge** | ⚡⚡ | ⭐⭐⭐ | Mix of Polylines and Reverse Projection |
| **Naive Interpolating** | ⚡⚡⚡ | ⭐⭐ | Fast interpolation, some stretching |
| **Naive** | ⚡⚡⚡ | ⭐⭐ | Fast fill, nearest pixel copy |
| **Reverse Projection** | ⚡⚡ | ⭐⭐⭐ | Works backward, leaves some gaps |
| **No Fill** | ⚡⚡⚡ | ⭐ | Shifts only, shows gaps |

**Recommended:** `Polylines Soft` for quality, `GPU Warp (Fast)` for speed.

## GPU Acceleration

Depth processing is **automatically GPU-accelerated** when CUDA is available:
- **5-20x faster** blur operations
- **Automatic fallback** to CPU if GPU unavailable
- **Zero configuration** - works out of the box

## Native VR Setup

### Requirements
1. Install PyOpenXR dependencies:
```bash
pip install -r requirements.txt
```

2. Install a VR runtime:
   - **SteamVR** (recommended) - Supports most headsets
   - **Oculus Runtime** - For Meta Quest headsets
   - **Windows Mixed Reality** - Built into Windows 10/11

3. Connect your VR headset

### Supported Headsets
- Meta Quest (1, 2, 3, Pro)
- HTC Vive / Vive Pro
- Valve Index
- Windows Mixed Reality headsets
- Any OpenXR-compatible device

### Troubleshooting VR
- Use **Native VR Status** node to check setup
- Ensure VR runtime is running before launching ComfyUI (will probably autostart though)
- Check that headset is properly connected
- Try restarting SteamVR/Oculus if viewer doesn't launch

## StereoDiffusion Setup

### Requirements
- CUDA-capable GPU with 8GB+ VRAM (16GB recommended)
- Python 3.8+
- PyTorch 2.0+

### First Run
- Downloads Stable Diffusion model (releaseversion SD1.5) (~4GB) if not cached
- Null-text optimization takes ~2-3 minutes on modern GPU
- Model is cached for faster subsequent runs

### Performance Tips
- Lower `num_ddim_steps` to 30 for faster processing
- Disable `null_text_optimization` for 3x speed (lower quality)
- Use `guidance_scale` 3-5 to reduce "burned" look

### Troubleshooting StereoDiffusion
- **Out of Memory**: Reduce `num_ddim_steps`, close other apps
- **Black Output**: Check depth map is valid grayscale
- **Poor Quality**: Enable `null_text_optimization`, increase `num_ddim_steps`

## License

MIT License - see [LICENSE](LICENSE) file for details.

**Note**: This project includes code from multiple sources:
- StereoDiffusion components are based on [StereoDiffusion](https://github.com/lez-s/StereoDiffusion) (MIT License)
- Diffusion utilities derived from [prompt-to-prompt](https://github.com/google/prompt-to-prompt) (Apache 2.0)
- See [NOTICE](NOTICE) file for full attribution

## Credits

Created by [Dobidop](https://github.com/Dobidop)

## Acknowledgments

### StereoDiffusion
```bibtex
@inproceedings{wang2024stereodiffusion,
  title={StereoDiffusion: Training-Free Stereo Image Generation Using Latent Diffusion Models},
  author={Wang, Lezhong and Frisvad, Jeppe Revall and Jensen, Mark Bo and Bigdeli, Siavash Arjomand},
  booktitle={CVPR},
  year={2024}
}
```

### Prompt-to-Prompt
```bibtex
@article{hertz2022prompt,
  title={Prompt-to-prompt image editing with cross attention control},
  author={Hertz, Amir and Mokady, Ron and Tenenbaum, Jay and Aberman, Kfir and Pritch, Yael and Cohen-Or, Daniel},
  year={2022}
}
```

## Contributing

Contributions welcome! Please submit issues or pull requests.

## Support

- **Issues**: [GitHub Issues](https://github.com/Dobidop/ComfyStereo/issues)

---
