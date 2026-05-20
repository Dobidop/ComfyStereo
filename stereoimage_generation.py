try:
    from numba import njit, prange
except Exception as e:
    print(f"WARNING! Numba failed to import! Stereoimage generation will be much slower! ({str(e)})")
    from builtins import range as prange
    def njit(parallel=False):
        def Inner(func): return lambda *args, **kwargs: func(*args, **kwargs)
        return Inner
import numpy as np
from PIL import Image
import math
from scipy.ndimage import convolve1d, sobel
import torch
import torch.nn.functional as F

# GPU availability check
CUDA_AVAILABLE = torch.cuda.is_available()

# ModernGL for mesh-based GPU rasterization via OpenGL
try:
    import moderngl
    MODERNGL_AVAILABLE = True
except ImportError:
    MODERNGL_AVAILABLE = False
    moderngl = None

# Cached ModernGL context (singleton)
_moderngl_ctx = None

# One-shot log so we know which GPU warp implementation is actually in use.
_warp_path_logged = False

def _get_moderngl_context():
    """Get or create a cached ModernGL standalone context."""
    global _moderngl_ctx
    if _moderngl_ctx is None:
        _moderngl_ctx = moderngl.create_standalone_context()
    return _moderngl_ctx


def _get_device(*tensors):
    """Get the best available device, with CUDA fallback to CPU on error."""
    for t in tensors:
        if t.is_cuda:
            return t.device
    if CUDA_AVAILABLE:
        try:
            torch.zeros(1, device='cuda')
            return torch.device('cuda')
        except RuntimeError:
            return torch.device('cpu')
    return torch.device('cpu')


def apply_stereo_divergence_gpu(image_tensor, depth_tensor, divergence_px, separation_px,
                                 stereo_offset_exponent, convergence_point=0.5):
    """
    GPU-accelerated stereo divergence using PyTorch grid_sample.
    Supports batched inputs for processing multiple frames simultaneously.

    Parameters:
        image_tensor (torch.Tensor): Input image [B, C, H, W], values 0-1
        depth_tensor (torch.Tensor): Depth map [B, H, W], values 0-1 (or 0-255, will be normalized)
        divergence_px (float): Divergence in pixels
        separation_px (float): Additional separation in pixels
        stereo_offset_exponent (float): Exponent for depth-to-offset mapping
        convergence_point (float): Which depth appears at screen plane (0.0-1.0)

    Returns:
        torch.Tensor: Warped image [B, C, H, W], values 0-1
    """
    device = _get_device(depth_tensor, image_tensor)
    image_tensor = image_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    B, _, H, W = image_tensor.shape

    # Normalize depth to 0-1 range if needed (per-image)
    depth_min = depth_tensor.amin(dim=(1, 2), keepdim=True)
    depth_max = depth_tensor.amax(dim=(1, 2), keepdim=True)
    needs_rescale = (depth_max > 1.0).any()
    if needs_rescale:
        depth_tensor = depth_tensor / 255.0
        depth_min = depth_tensor.amin(dim=(1, 2), keepdim=True)
        depth_max = depth_tensor.amax(dim=(1, 2), keepdim=True)

    # Normalize depth map to 0-1 range per image
    depth_range = depth_max - depth_min
    normalized_depth = torch.where(
        depth_range > 1e-6,
        (depth_tensor - depth_min) / depth_range.clamp(min=1e-6),
        torch.zeros_like(depth_tensor)
    )

    # Apply convergence point offset and exponent
    normalized_depth = normalized_depth - convergence_point
    sign = torch.sign(normalized_depth)
    abs_depth = torch.abs(normalized_depth)
    offset_depth = sign * torch.pow(abs_depth, stereo_offset_exponent)

    # Calculate pixel offsets [B, H, W]
    pixel_offset = offset_depth * divergence_px + separation_px
    offset_normalized = pixel_offset / (W / 2)

    # Create sampling grid [B, H, W, 2]
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    # Expand to batch: [B, H, W]
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    grid_x_warped = grid_x - offset_normalized
    grid = torch.stack([grid_x_warped, grid_y], dim=-1)  # [B, H, W, 2]

    # grid_sample: [B, C, H, W] x [B, H, W, 2] -> [B, C, H, W]
    warped = F.grid_sample(
        image_tensor, grid,
        mode='bilinear', padding_mode='border', align_corners=True
    )

    return warped


def warp_and_fill_gpu(image_tensor, depth_tensor, divergence_px, separation_px,
                      stereo_offset_exponent, convergence_point=0.5):
    """
    GPU stereo warp with built-in edge-stretching fill.

    Instead of warping and then post-filling gaps, this modifies the warp grid
    at gap positions to smoothly interpolate between border source positions.
    The result is a natural "edge stretching" effect: the foreground edge pixels
    are stretched across the disocclusion gap via grid_sample on the original image.

    This avoids all bilinear smearing artifacts because gap pixels sample from
    interpolated source positions in the original (pre-warp) image, not from
    the warped image.

    Args:
        image_tensor: [B, C, H, W] input image, values 0-1
        depth_tensor: [B, H, W] depth map (0-255 or 0-1)
        divergence_px: horizontal shift in pixels
        separation_px: additional separation in pixels
        stereo_offset_exponent: power curve for depth mapping
        convergence_point: which depth maps to zero offset

    Returns:
        warped: [B, C, H, W] warped image with gaps filled via edge stretching
        gap_mask: [B, H, W] bool tensor marking gap/dilated locations
    """
    device = _get_device(depth_tensor, image_tensor)
    image_tensor = image_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    B, _, H, W = image_tensor.shape

    # --- Compute pixel offset from depth ---
    d = depth_tensor.clone()
    d_max_all = d.amax(dim=(1, 2), keepdim=True)
    if (d_max_all > 1.0).any():
        d = d / 255.0
    d_min = d.amin(dim=(1, 2), keepdim=True)
    d_max = d.amax(dim=(1, 2), keepdim=True)
    d_range = d_max - d_min
    normalized_depth = torch.where(
        d_range > 1e-6,
        (d - d_min) / d_range.clamp(min=1e-6),
        torch.zeros_like(d)
    )

    depth_shifted = normalized_depth - convergence_point
    sign = torch.sign(depth_shifted)
    abs_depth = torch.abs(depth_shifted)
    offset_depth = sign * torch.pow(abs_depth, stereo_offset_exponent)

    pixel_offset = offset_depth * divergence_px + separation_px  # [B, H, W]

    # --- Compute gap mask via forward mapping ---
    col_indices = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(B, H, W)
    dest_cols = (col_indices + pixel_offset).long()

    valid_dest = (dest_cols >= 0) & (dest_cols < W)
    dest_clamped = dest_cols.clamp(0, W - 1)

    hit_count = torch.zeros(B, H, W, device=device)
    hit_count.scatter_add_(2, dest_clamped, valid_dest.float())
    gap_mask = hit_count < 0.5

    # Dilate at depth edges (ghosting fix)
    offset_grad = torch.abs(pixel_offset[:, :, 1:] - pixel_offset[:, :, :-1])
    depth_edge = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    depth_edge[:, :, :-1] = offset_grad > 1.5
    depth_edge[:, :, 1:] = depth_edge[:, :, 1:] | (offset_grad > 1.5)

    dilated = gap_mask.clone()
    dilated[:, :, 1:] = dilated[:, :, 1:] | (gap_mask[:, :, :-1] & depth_edge[:, :, 1:])
    dilated[:, :, :-1] = dilated[:, :, :-1] | (gap_mask[:, :, 1:] & depth_edge[:, :, :-1])
    gap_mask = dilated

    # --- Create warp grid ---
    offset_normalized = pixel_offset / (W / 2)

    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_x = grid_x.unsqueeze(0).expand(B, -1, -1)
    grid_y = grid_y.unsqueeze(0).expand(B, -1, -1)

    grid_x_warped = grid_x - offset_normalized

    # --- Edge stretching: interpolate warp grid at gap positions ---
    # Instead of sampling from invalid positions, gap pixels sample from
    # smoothly interpolated source positions between the fg and bg borders.
    if gap_mask.any():
        valid_pixels = ~gap_mask
        cols = torch.arange(W, device=device, dtype=torch.long).unsqueeze(0).unsqueeze(0).expand(B, H, W)

        # Find nearest valid pixel on left (cummax)
        left_valid_col = torch.where(valid_pixels, cols, torch.full_like(cols, -1))
        left_nearest, _ = torch.cummax(left_valid_col, dim=2)
        left_dist = (cols - left_nearest).float()
        has_left = left_nearest >= 0

        # Find nearest valid pixel on right (reverse cummax)
        right_valid_flip = torch.where(
            torch.flip(valid_pixels, [2]),
            torch.flip(cols, [2]),
            torch.full_like(cols, -1)
        )
        right_nearest_flip, _ = torch.cummax(right_valid_flip, dim=2)
        right_nearest = torch.flip(right_nearest_flip, [2])
        right_dist = (right_nearest - cols).float()
        has_right = right_nearest >= 0

        # Edge stretching: magnify a few valid border pixels across each gap half.
        # Instead of interpolating between distant source positions (which sweeps
        # through the original image), each half of the gap samples from a small
        # range of actual valid warped grid values near its border — stretching
        # those few pixels to fill the gap like a rubber band.
        total_dist = torch.clamp(left_dist + right_dist, min=1.0)
        half_gap = total_dist * 0.5

        # Number of valid border pixels to stretch across each half-gap
        stretch_pixels = 3

        # Left side: gather grid_x at border and K pixels deeper into valid region
        left_base_grid = grid_x_warped.gather(2, left_nearest.clamp(0, W - 1))
        left_deep_grid = grid_x_warped.gather(2, (left_nearest - stretch_pixels).clamp(0, W - 1))
        # Map left gap distance [0, half_gap] to source range [border, border-K]
        left_t = torch.clamp(left_dist / half_gap, 0.0, 1.0)
        left_stretch = left_base_grid * (1.0 - left_t) + left_deep_grid * left_t

        # Right side: gather grid_x at border and K pixels deeper into valid region
        right_base_grid = grid_x_warped.gather(2, right_nearest.clamp(0, W - 1))
        right_deep_grid = grid_x_warped.gather(2, (right_nearest + stretch_pixels).clamp(0, W - 1))
        # Map right gap distance [0, half_gap] to source range [border, border+K]
        right_t = torch.clamp(right_dist / half_gap, 0.0, 1.0)
        right_stretch = right_base_grid * (1.0 - right_t) + right_deep_grid * right_t

        # Blend between left-stretched and right-stretched in the middle zone
        t = left_dist / total_dist  # 0 at left border, 1 at right border
        t = torch.where(~has_left, torch.ones_like(t), t)
        t = torch.where(~has_right, torch.zeros_like(t), t)
        blend = torch.clamp((t - 0.35) / 0.3, 0.0, 1.0)
        blend = blend * blend * (3.0 - 2.0 * blend)  # smoothstep

        gap_grid_x = left_stretch * (1.0 - blend) + right_stretch * blend
        grid_x_warped = torch.where(gap_mask, gap_grid_x, grid_x_warped)

    # --- Single grid_sample with filled grid ---
    grid = torch.stack([grid_x_warped, grid_y], dim=-1)  # [B, H, W, 2]
    warped = F.grid_sample(
        image_tensor, grid,
        mode='bilinear', padding_mode='border', align_corners=True
    )

    return warped, gap_mask


def forward_warp_gpu(image_tensor, depth_tensor, divergence_px, separation_px,
                     stereo_offset_exponent, convergence_point=0.5,
                     gradient_threshold=1.5, max_stretch=8):
    """
    GPU forward-mapping stereo warp with gradient-aware connectivity.

    Forward-maps each source pixel to its destination. Adjacent source pixels with
    similar offsets are "connected" — the output range between their destinations is
    filled by interpolation (like mesh triangles stretching). At depth discontinuities,
    connectivity breaks and gaps appear naturally.

    The forward map is then inverted to build a source-coordinate grid for grid_sample,
    which renders the final image from the original (unwarped) source with clean
    bilinear interpolation.

    Args:
        image_tensor: [B, C, H, W] input image, values 0-1
        depth_tensor: [B, H, W] depth map (0-255 or 0-1)
        divergence_px: horizontal shift in pixels
        separation_px: additional constant separation in pixels
        stereo_offset_exponent: power curve for depth-to-offset mapping
        convergence_point: which depth maps to zero offset (0.0-1.0)
        gradient_threshold: max offset difference between adjacent pixels to be connected
        max_stretch: max output pixels a single source pair can span

    Returns:
        warped: [B, C, H, W] warped image with clean bilinear colors
        gap_mask: [B, H, W] bool tensor, True = disocclusion gap
    """
    device = _get_device(depth_tensor, image_tensor)
    image_tensor = image_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    B, _, H, W = image_tensor.shape

    # --- Step 1: Compute pixel offset from depth ---
    d = depth_tensor.clone()
    d_max_all = d.amax(dim=(1, 2), keepdim=True)
    if (d_max_all > 1.0).any():
        d = d / 255.0
    d_min = d.amin(dim=(1, 2), keepdim=True)
    d_max = d.amax(dim=(1, 2), keepdim=True)
    d_range = d_max - d_min
    normalized_depth = torch.where(
        d_range > 1e-6,
        (d - d_min) / d_range.clamp(min=1e-6),
        torch.zeros_like(d)
    )

    depth_shifted = normalized_depth - convergence_point
    sign = torch.sign(depth_shifted)
    abs_depth = torch.abs(depth_shifted)
    offset_depth = sign * torch.pow(abs_depth, stereo_offset_exponent)

    pixel_offset = offset_depth * divergence_px + separation_px  # [B, H, W]

    # --- Step 2: Compute forward destinations (sub-pixel float) ---
    col_float = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)
    dest = col_float + pixel_offset  # [B, H, W]

    # --- Step 3: Determine connectivity ---
    # Connected if offset difference between adjacent pixels is below threshold
    offset_diff = torch.abs(pixel_offset[:, :, 1:] - pixel_offset[:, :, :-1])
    connected = offset_diff < gradient_threshold  # [B, H, W-1]

    # --- Step 4: Build inverse source map via fixed-iteration scatter with z-buffer ---
    source_map = torch.full((B, H, W), -1.0, device=device, dtype=torch.float32)
    z_buffer = torch.full((B, H, W), -1.0, device=device, dtype=torch.float32)

    # Paired values for connected segments [B, H, W-1]
    dest_left = dest[:, :, :-1]
    dest_right = dest[:, :, 1:]
    depth_left = normalized_depth[:, :, :-1]
    depth_right = normalized_depth[:, :, 1:]

    # Start of output range: minimum destination of each pair
    dest_min = torch.min(dest_left, dest_right)
    floor_start = torch.floor(dest_min).long()

    # Source column indices for the left pixel in each pair
    src_col_base = torch.arange(W - 1, device=device, dtype=torch.float32).view(1, 1, W - 1).expand(B, H, W - 1)

    # Segment width (can be negative for reversed segments)
    segment_width = dest_right - dest_left
    safe_width = torch.where(segment_width.abs() < 1e-4,
                             torch.ones_like(segment_width),
                             segment_width)

    for k in range(max_stretch):
        c = floor_start + k  # [B, H, W-1] target output column
        c_safe = c.clamp(0, W - 1)

        # Fractional position within the segment: 0 at dest_left, 1 at dest_right
        # No +0.5: pixel centers are at integer positions (align_corners=True)
        frac = (c.float() - dest_left) / safe_width

        # Half-open [0, 1): each pair owns up to but not including the next pair's start
        # This prevents two pairs from claiming the same output column
        valid = connected & (c >= 0) & (c < W) & (frac >= 0.0) & (frac < 1.0)

        # Interpolated source position and depth
        src_pos = src_col_base + frac
        interp_depth = depth_left * (1.0 - frac) + depth_right * frac

        # Z-buffer: only write if this depth is closer (higher = closer to camera)
        current_z = z_buffer.gather(2, c_safe)
        current_src = source_map.gather(2, c_safe)

        better = valid & (interp_depth > current_z + 1e-6)

        new_z = torch.where(better, interp_depth, current_z)
        new_src = torch.where(better, src_pos, current_src)

        z_buffer.scatter_(2, c_safe, new_z)
        source_map.scatter_(2, c_safe, new_src)

    # --- Step 5: Fill disocclusion gaps ---
    unfilled = source_map < 0  # [B, H, W] — the gap mask
    cols = torch.arange(W, device=device, dtype=torch.long).view(1, 1, W).expand(B, H, W)
    filled_mask = ~unfilled

    # Find nearest filled pixel on left
    left_col = torch.where(filled_mask, cols, torch.full_like(cols, -1))
    left_nearest, _ = torch.cummax(left_col, dim=2)
    has_left = left_nearest >= 0

    # Find nearest filled pixel on right
    right_col_flip = torch.where(
        torch.flip(filled_mask, [2]),
        torch.flip(cols, [2]),
        torch.full_like(cols, -1)
    )
    right_nearest_flip, _ = torch.cummax(right_col_flip, dim=2)
    right_nearest = torch.flip(right_nearest_flip, [2])
    has_right = right_nearest >= 0

    # Gather source positions and depths at gap borders
    left_src = source_map.gather(2, left_nearest.clamp(0, W - 1))
    right_src = source_map.gather(2, right_nearest.clamp(0, W - 1))
    left_z = z_buffer.gather(2, left_nearest.clamp(0, W - 1))
    right_z = z_buffer.gather(2, right_nearest.clamp(0, W - 1))

    # Linear interpolation parameter
    left_dist = (cols - left_nearest).float()
    right_dist = (right_nearest - cols).float()
    total_dist = (left_dist + right_dist).clamp(min=1.0)
    t = left_dist / total_dist  # 0 at left border, 1 at right border

    t = torch.where(~has_left, torch.ones_like(t), t)
    t = torch.where(~has_right, torch.zeros_like(t), t)

    # Background bias: favor the side with lower depth (background)
    left_is_bg = left_z < right_z
    t_biased = torch.where(left_is_bg,
                           1.0 - (1.0 - t).pow(0.5),     # push toward left (bg): smaller t
                           t.pow(0.5))                    # push toward right (bg): larger t

    gap_src = left_src * (1.0 - t_biased) + right_src * t_biased
    source_map = torch.where(unfilled & (has_left | has_right), gap_src, source_map)

    # Clamp any remaining unfilled pixels
    source_map = source_map.clamp(0, W - 1)

    # --- Step 6: Convert source map to grid_sample grid ---
    grid_x = source_map * 2.0 / (W - 1) - 1.0
    grid_y = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(B, H, W)

    grid = torch.stack([grid_x, grid_y], dim=-1)  # [B, H, W, 2]
    warped = F.grid_sample(
        image_tensor, grid,
        mode='bilinear', padding_mode='border', align_corners=True
    )

    return warped, unfilled, normalized_depth


def forward_warp_mesh(image_tensor, depth_tensor, divergence_px, separation_px,
                      stereo_offset_exponent, convergence_point=0.5,
                      gradient_threshold=1.5, max_stretch=8,
                      fill_depth_threshold=0.2, edge_setback=4,
                      extend_radius=8, extend_edge_threshold=0.1,
                      extend_fade_distance=20.0):
    """
    GPU mesh-based stereo warp using ModernGL OpenGL rasterization.
    Drop-in replacement for forward_warp_gpu().

    Builds a triangle mesh from the depth map (each pixel = vertex), removes
    triangles at depth discontinuities, displaces vertices by the stereo offset,
    and rasterizes with hardware z-buffering for clean, artifact-free results.

    Args:
        image_tensor: [B, C, H, W] input image, values 0-1
        depth_tensor: [B, H, W] depth map (0-255 or 0-1)
        divergence_px: horizontal shift in pixels
        separation_px: additional constant separation in pixels
        stereo_offset_exponent: power curve for depth-to-offset mapping
        convergence_point: which depth maps to zero offset (0.0-1.0)
        gradient_threshold: max offset difference between triangle vertices to keep
        max_stretch: unused (kept for API compatibility)

    Returns:
        warped: [B, C, H, W] warped image
        gap_mask: [B, H, W] bool tensor, True = disocclusion gap
    """
    device = _get_device(depth_tensor, image_tensor)
    image_tensor = image_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    B, C, H, W = image_tensor.shape
    N_verts = H * W

    # --- Step 1: Compute pixel offset from depth (identical to forward_warp_gpu) ---
    d = depth_tensor.clone()
    d_max_all = d.amax(dim=(1, 2), keepdim=True)
    if (d_max_all > 1.0).any():
        d = d / 255.0
    d_min = d.amin(dim=(1, 2), keepdim=True)
    d_max = d.amax(dim=(1, 2), keepdim=True)
    d_range = d_max - d_min
    normalized_depth = torch.where(
        d_range > 1e-6,
        (d - d_min) / d_range.clamp(min=1e-6),
        torch.zeros_like(d)
    )

    # Silhouette extension ("fatten"): propagate fg depth into adjacent bg
    # pixels on the disocclusion side, so the warped fg covers what would
    # otherwise be a 1-N px gap at the silhouette. Many depth-map generators
    # systematically under-estimate fg extent on bg->fg edges; without this
    # those mislabeled pixels lag behind the warped fg and produce a "bite" /
    # "translated edge" artifact.
    #
    # Direction is set by the warp direction, so only the disocclusion side is
    # thickened (never the occlusion side).
    #
    # The trigger ("edge threshold") measures local sharpness: max adjacent-
    # pixel depth diff within the search window. A 1-px crisp edge yields
    # sharpness ≈ 1; a 5-px AA ramp ≈ 0.2; a smooth gradient is small. Smooth
    # surfaces never trigger, so the fatten is silhouette-only.
    #
    # The candidate value at offset k is `shifted_depth - k * decay`, with
    # HOLD pixels right next to the silhouette held at full strength before
    # the linear taper begins. Taking the max over k yields a smooth ramp
    # from fg toward bg in the depth output.
    #
    # Params (all from user-facing units):
    #   extend_radius (px)          - max reach
    #   extend_edge_threshold (0-1) - min sharpness to trigger
    #   extend_fade_distance (px)   - 1/decay; how far the taper fades
    _FATTEN_RADIUS = int(extend_radius)
    _FATTEN_DECAY = 1.0 / max(float(extend_fade_distance), 1.0)
    _FATTEN_JUMP = float(extend_edge_threshold)
    _FATTEN_HOLD = 1
    if _FATTEN_RADIUS > 0:
        shift_dir = 1 if divergence_px < 0 else -1

        # Sharpness probe: max adjacent-pixel diff anywhere in the disocclusion
        # window. A 1-px sharp edge produces sharpness ≈ 1; a 5-px AA ramp
        # produces sharpness ≈ 0.2; smooth gradients produce ≈ small step.
        step_left  = (normalized_depth - torch.roll(normalized_depth, shifts=1, dims=2)).abs()
        step_right = (normalized_depth - torch.roll(normalized_depth, shifts=-1, dims=2)).abs()
        local_step = torch.maximum(step_left, step_right)
        window_sharpness = local_step.clone()
        for k in range(1, _FATTEN_RADIUS + 1):
            window_sharpness = torch.maximum(
                window_sharpness,
                torch.roll(local_step, shifts=shift_dir * k, dims=2)
            )
        behind_real_silhouette = window_sharpness > _FATTEN_JUMP

        fattened_depth = normalized_depth.clone()
        for k in range(1, _FATTEN_RADIUS + 1):
            shifted_depth = torch.roll(normalized_depth, shifts=shift_dir * k, dims=2)
            decay_steps = max(0, k - _FATTEN_HOLD)
            candidate = shifted_depth - _FATTEN_DECAY * float(decay_steps)
            candidate = torch.where(behind_real_silhouette, candidate, fattened_depth)
            fattened_depth = torch.maximum(fattened_depth, candidate)
    else:
        fattened_depth = normalized_depth
    normalized_depth = fattened_depth

    fattened_depth_out = normalized_depth.clone()  # exposed for the node's debug output

    depth_shifted = normalized_depth - convergence_point
    sign = torch.sign(depth_shifted)
    abs_depth = torch.abs(depth_shifted)
    offset_depth = sign * torch.pow(abs_depth, stereo_offset_exponent)

    pixel_offset = offset_depth * divergence_px + separation_px  # [B, H, W]

    # --- Step 2: Build triangle indices (vectorized, on GPU then move to CPU) ---
    y_idx = torch.arange(H - 1, device=device)
    x_idx = torch.arange(W - 1, device=device)
    yy, xx = torch.meshgrid(y_idx, x_idx, indexing='ij')  # [H-1, W-1]
    yy_f = yy.reshape(-1)
    xx_f = xx.reshape(-1)

    v00 = yy_f * W + xx_f
    v10 = yy_f * W + (xx_f + 1)
    v01 = (yy_f + 1) * W + xx_f
    v11 = (yy_f + 1) * W + (xx_f + 1)

    tri_a = torch.stack([v00, v10, v01], dim=1)
    tri_b = torch.stack([v11, v10, v01], dim=1)
    all_tris = torch.cat([tri_a, tri_b], dim=0)  # [2*(H-1)*(W-1), 3]

    # --- Step 3: Gradient-based triangle culling ---
    offset_flat = pixel_offset.reshape(B, -1)

    v0_off = offset_flat[:, all_tris[:, 0]]
    v1_off = offset_flat[:, all_tris[:, 1]]
    v2_off = offset_flat[:, all_tris[:, 2]]

    max_diff = torch.max(
        torch.max(torch.abs(v0_off - v1_off), torch.abs(v0_off - v2_off)),
        torch.abs(v1_off - v2_off)
    )

    # Depth-gradient cull: independent of divergence. A real fg/bg edge has
    # a large normalized-depth jump between vertices even when the resulting
    # offset diff happens to fall under gradient_threshold (low divergence or
    # shallow depth contrast). Without this, those triangles survive the cull
    # and rasterize barycentric fg-tinted color into the disocclusion zone as
    # a thin "translated edge" 1-3 px wide.
    _DEPTH_GRADIENT_THRESHOLD = 0.1
    depth_flat = normalized_depth.reshape(B, -1)
    d0 = depth_flat[:, all_tris[:, 0]]
    d1 = depth_flat[:, all_tris[:, 1]]
    d2 = depth_flat[:, all_tris[:, 2]]
    max_depth_diff = torch.max(
        torch.max(torch.abs(d0 - d1), torch.abs(d0 - d2)),
        torch.abs(d1 - d2)
    )

    # Keep triangle if it passes BOTH criteria in ANY batch item
    keep = ((max_diff < gradient_threshold) &
            (max_depth_diff < _DEPTH_GRADIENT_THRESHOLD)).any(dim=0)
    tri_filtered = all_tris[keep].contiguous()

    # Convert indices to numpy for OpenGL
    indices_np = tri_filtered.cpu().numpy().astype(np.int32)

    # --- Step 4: Compute vertex positions ---
    col_float = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)
    row_float = torch.arange(H, device=device, dtype=torch.float32).view(1, H, 1).expand(B, H, W)

    dest_x = col_float + pixel_offset
    dest_y = row_float

    # Convert to clip space [-1, 1] for OpenGL
    clip_x = dest_x / (W - 1) * 2.0 - 1.0
    clip_y = -(dest_y / (H - 1) * 2.0 - 1.0)  # flip Y for OpenGL
    # Map depth to NDC z range [-0.99, 0.99]: closer objects get smaller z (wins '<' test)
    # Use [-.99, .99] instead of [-1, 1] to avoid clipping at the exact clip planes
    clip_z = (1.0 - normalized_depth) * 1.98 - 0.99

    # Per-vertex colors [B, H*W, C]
    vertex_colors = image_tensor.permute(0, 2, 3, 1).reshape(B, N_verts, C)

    # Per-vertex depth for gap fill (so we know rendered depth at each output pixel)
    vertex_depth = normalized_depth.reshape(B, N_verts, 1)  # [B, H*W, 1]

    # --- Step 5: Rasterize each batch item with ModernGL ---
    ctx = _get_moderngl_context()

    # Shader: pass position, color, and depth as vertex attributes
    # Output color+depth to color texture, coverage flag to flag texture
    prog = ctx.program(
        vertex_shader='''
            #version 330
            in vec3 in_position;
            in vec3 in_color;
            in float in_depth;
            out vec3 v_color;
            out float v_depth;
            void main() {
                gl_Position = vec4(in_position.xy, in_position.z, 1.0);
                v_color = in_color;
                v_depth = in_depth;
            }
        ''',
        fragment_shader='''
            #version 330
            in vec3 v_color;
            in float v_depth;
            layout(location = 0) out vec4 fragColor;
            layout(location = 1) out float fragFlag;
            void main() {
                fragColor = vec4(v_color, v_depth);
                fragFlag = 1.0;
            }
        ''',
    )

    # Create index buffer (shared across batch — same triangle topology)
    ibo = ctx.buffer(indices_np.tobytes())

    # Create framebuffer attachments
    color_tex = ctx.texture((W, H), 4, dtype='f4')
    flag_tex = ctx.texture((W, H), 1, dtype='f4')
    depth_rb = ctx.depth_renderbuffer((W, H))
    fbo = ctx.framebuffer(
        color_attachments=[color_tex, flag_tex],
        depth_attachment=depth_rb
    )

    results = []
    depths = []
    masks = []

    for b in range(B):
        # Build interleaved vertex data: [x, y, z, r, g, b, depth] per vertex
        positions = torch.stack([
            clip_x[b].reshape(-1),
            clip_y[b].reshape(-1),
            clip_z[b].reshape(-1),
        ], dim=1)  # [N_verts, 3]
        colors = vertex_colors[b]  # [N_verts, C]
        vdepth = vertex_depth[b]   # [N_verts, 1]

        vertex_data = torch.cat([positions, colors, vdepth], dim=1)  # [N_verts, 7]
        vertex_np = vertex_data.cpu().numpy().astype(np.float32)

        vbo = ctx.buffer(vertex_np.tobytes())
        vao = ctx.vertex_array(prog, [(vbo, '3f 3f 1f', 'in_position', 'in_color', 'in_depth')], ibo)

        fbo.use()
        ctx.enable(moderngl.DEPTH_TEST)
        ctx.depth_func = '<'  # smaller z = closer to camera = wins depth test
        fbo.clear(red=0.0, green=0.0, blue=0.0, alpha=0.0, depth=1.0)
        vao.render(moderngl.TRIANGLES)

        # Read back results
        color_data = np.frombuffer(color_tex.read(), dtype=np.float32).reshape(H, W, 4)
        flag_data = np.frombuffer(flag_tex.read(), dtype=np.float32).reshape(H, W)

        # OpenGL renders bottom-up, flip to top-down
        color_data = color_data[::-1].copy()
        flag_data = flag_data[::-1].copy()

        results.append(color_data[:, :, :3])  # RGB
        depths.append(color_data[:, :, 3])     # Rendered depth in alpha channel
        masks.append(flag_data < 0.5)           # True = gap (unfilled)

        vbo.release()
        vao.release()

    # Clean up OpenGL resources
    ibo.release()
    color_tex.release()
    flag_tex.release()
    depth_rb.release()
    fbo.release()
    prog.release()

    # Convert results back to torch tensors
    warped_np = np.stack(results, axis=0)  # [B, H, W, 3]
    gap_mask_np = np.stack(masks, axis=0)  # [B, H, W]
    rendered_depth_np = np.stack(depths, axis=0)  # [B, H, W] — fg=high, bg=low (normalized)

    warped = torch.from_numpy(warped_np).permute(0, 3, 1, 2).to(device)  # [B, C, H, W]
    gap_mask = torch.from_numpy(gap_mask_np).to(device)  # [B, H, W]
    rendered_depth = torch.from_numpy(rendered_depth_np).to(device)  # [B, H, W]

    # --- Step 6: Depth-aware gap fill ---
    # Smear from the correct side based on eye, but prefer source pixels with
    # background-like depth. Otherwise the cummax-nearest-filled-pixel heuristic
    # picks the trailing edge of a nearby foreground object (e.g. another hair
    # strand across the disocclusion) and smears foreground color into what
    # should be background, since the actual bg source for the disocclusion lies
    # further away across that foreground.
    #
    # Logic: cummax twice — once over "filled & depth-below-threshold" (preferred
    # bg source), once over plain "filled" (fallback). Use the bg-restricted
    # source where one exists in the row, fall back to nearest-filled otherwise.
    # The fallback preserves prior behavior in rows where the depth-restricted
    # search finds nothing (e.g. frame edges, all-foreground rows).
    # Extend the gap mask to include "soft gap" transition pixels next to real gaps.
    # Kept stretched-mesh triangles at the fg edge (cells whose offset gradient
    # squeaks under gradient_threshold) render barycentric-interpolated color and
    # mid-range rendered_depth. These look like a thin "translated copy" of the
    # fg edge in the disocclusion zone. Treating them as gap and filling from bg
    # cleans up that ghost edge without touching legitimate fg-bg interface
    # content far from any gap. The original gap_mask is preserved for the
    # return value (so the node's MASK output still reflects only rasterizer-
    # uncovered pixels); only the local effective_gap_mask used by fill logic
    # is extended.
    #
    # Detection: inverse-warp consistency. For every rasterized pixel, compute
    # what source column its RENDERED depth would have shifted from (using the
    # same offset formula the forward warp used), then sample the actual SOURCE
    # depth at that column. A legitimate pixel matches: its rendered depth came
    # from a real surface that lives at the implied source location. A ghost
    # pixel from a stretched-mesh triangle does not match: its rendered depth
    # is barycentric-interpolated between fg and bg vertices and corresponds to
    # no real source surface at the implied column. This catches both hard-edge
    # ghosts (1-3 px bands) and soft / hair-edge ghosts (4+ px) without biting
    # legitimate fg silhouettes, where rendered_depth equals the actual fg depth
    # at the corresponding source column.
    rd_shifted = rendered_depth - convergence_point
    rd_sign = torch.sign(rd_shifted)
    rd_abs = torch.abs(rd_shifted)
    rd_offset_depth = rd_sign * torch.pow(rd_abs, stereo_offset_exponent)
    rd_pixel_offset = rd_offset_depth * divergence_px + separation_px  # [B, H, W]

    cols_f = torch.arange(W, device=device, dtype=torch.float32).view(1, 1, W).expand(B, H, W)
    src_col_implied = cols_f - rd_pixel_offset  # [B, H, W]
    grid_x_check = src_col_implied * 2.0 / max(W - 1, 1) - 1.0
    grid_y_check = torch.linspace(-1, 1, H, device=device).view(1, H, 1).expand(B, H, W)
    grid_check = torch.stack([grid_x_check, grid_y_check], dim=-1)  # [B, H, W, 2]
    src_depth_at_implied = F.grid_sample(
        normalized_depth.unsqueeze(1), grid_check,
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(1)  # [B, H, W]

    _CONSISTENCY_TOLERANCE = 0.15
    inconsistent = (torch.abs(src_depth_at_implied - rendered_depth) > _CONSISTENCY_TOLERANCE) & (~gap_mask)
    effective_gap_mask = gap_mask | inconsistent

    if effective_gap_mask.any():
        filled_mask = ~effective_gap_mask
        # Bg-eligibility for fill sources. Lower fill_depth_threshold = stricter
        # (only deep-bg pixels qualify). Mid-depth pixels at the fg/bg boundary
        # are barycentric-interpolated by stretched-mesh triangles and tend to
        # be fg-tinted; excluding them prevents the "translated fg edge" smear.
        bg_eligible = filled_mask & (rendered_depth < fill_depth_threshold)

        # Fg-barrier mask: pixels clearly belonging to a foreground object.
        # The bg search treats these as blockers (it will NOT reach past them)
        # so a thin fg silhouette can't be skipped over to sample bg from the
        # wrong side of it — that was producing intermittent fg→bg smears in
        # small gaps adjacent to thin fg features. 0.7 is a generous threshold:
        # anything above it is unambiguously fg.
        fg_barrier_threshold = 0.7
        fg_barrier = filled_mask & (rendered_depth >= fg_barrier_threshold)

        # Cap how far the gap-fill is allowed to reach for a bg source.
        max_fill_dist = max(int(abs(divergence_px) * 1.5), 40)

        cols = torch.arange(W, device=device, dtype=torch.long).view(1, 1, W).expand(B, H, W)
        SENT_L = -1       # left-search sentinel ( cummax  → "none found")
        SENT_R = W        # right-search sentinel ( cummin → "none found")

        # --- LEFT search (largest col ≤ current; smaller col = further left) ---
        bg_col_L   = torch.where(bg_eligible, cols, torch.full_like(cols, SENT_L))
        fg_col_L   = torch.where(fg_barrier,  cols, torch.full_like(cols, SENT_L))
        fill_col_L = torch.where(filled_mask, cols, torch.full_like(cols, SENT_L))
        bg_near_L,   _ = torch.cummax(bg_col_L,   dim=2)
        fg_near_L,   _ = torch.cummax(fg_col_L,   dim=2)
        fill_near_L, _ = torch.cummax(fill_col_L, dim=2)
        # Primary bg requires barrier-respect: no fg between current and bg.
        bg_valid_L  = (bg_near_L >= 0) & (bg_near_L >= fg_near_L) & ((cols - bg_near_L) <= max_fill_dist)
        fill_has_L  = fill_near_L >= 0
        bg_sample_L = (bg_near_L - edge_setback).clamp(min=0)
        # Fallback sample: step edge_setback past the gap boundary into whatever
        # is there. Does NOT consult bg_eligible — used for frame-edge gaps
        # where the bg-eligible search finds nothing usable.
        fallback_sample_L = (fill_near_L + edge_setback).clamp(min=0)

        # --- RIGHT search (smallest col ≥ current; larger col = further right) ---
        bg_col_R   = torch.where(bg_eligible, cols, torch.full_like(cols, SENT_R))
        fg_col_R   = torch.where(fg_barrier,  cols, torch.full_like(cols, SENT_R))
        fill_col_R = torch.where(filled_mask, cols, torch.full_like(cols, SENT_R))
        bg_near_R   = torch.flip(torch.cummin(torch.flip(bg_col_R,   [2]), dim=2).values, [2])
        fg_near_R   = torch.flip(torch.cummin(torch.flip(fg_col_R,   [2]), dim=2).values, [2])
        fill_near_R = torch.flip(torch.cummin(torch.flip(fill_col_R, [2]), dim=2).values, [2])
        bg_valid_R  = (bg_near_R < W) & (bg_near_R <= fg_near_R) & ((bg_near_R - cols) <= max_fill_dist)
        fill_has_R  = fill_near_R < W
        bg_sample_R = (bg_near_R + edge_setback).clamp(max=W - 1)
        fallback_sample_R = (fill_near_R - edge_setback).clamp(max=W - 1)

        # Per-eye primary/secondary direction assignment.
        if divergence_px >= 0:
            primary_bg_sample,   primary_bg_valid   = bg_sample_L, bg_valid_L
            primary_fill_has                        = fill_has_L
            primary_fallback                        = fallback_sample_L
            primary_fill_near                       = fill_near_L
            secondary_bg_sample, secondary_bg_valid = bg_sample_R, bg_valid_R
            secondary_fill_has                      = fill_has_R
            secondary_fallback                      = fallback_sample_R
            secondary_fill_near                     = fill_near_R
            bg_near_P_fallback                      = bg_near_L
            bg_near_S_fallback                      = bg_near_R
        else:
            primary_bg_sample,   primary_bg_valid   = bg_sample_R, bg_valid_R
            primary_fill_has                        = fill_has_R
            primary_fallback                        = fallback_sample_R
            primary_fill_near                       = fill_near_R
            secondary_bg_sample, secondary_bg_valid = bg_sample_L, bg_valid_L
            secondary_fill_has                      = fill_has_L
            secondary_fallback                      = fallback_sample_L
            secondary_fill_near                     = fill_near_L
            bg_near_P_fallback                      = bg_near_R
            bg_near_S_fallback                      = bg_near_L

        # Safety #1: confirm primary/secondary BG samples are bg-eligible.
        # If setback overshoots into a gap or fg pixel, fall back to the
        # un-stepped nearest bg column.
        sample_is_bg_P = bg_eligible.gather(2, primary_bg_sample.clamp(0, W - 1))
        sample_is_bg_S = bg_eligible.gather(2, secondary_bg_sample.clamp(0, W - 1))
        primary_bg_sample   = torch.where(sample_is_bg_P, primary_bg_sample,
                                          bg_near_P_fallback.clamp(0, W - 1))
        secondary_bg_sample = torch.where(sample_is_bg_S, secondary_bg_sample,
                                          bg_near_S_fallback.clamp(0, W - 1))

        # Safety #2: the "fixed-step inward" fallback samples can land inside
        # the gap region itself (cleared to black). Verify each fallback column
        # is NOT in effective_gap_mask; if it is, snap back to the gap-boundary
        # fill_near (guaranteed non-gap by construction).
        primary_fallback_safe   = torch.where(
            effective_gap_mask.gather(2, primary_fallback.clamp(0, W - 1)),
            primary_fill_near.clamp(0, W - 1),
            primary_fallback.clamp(0, W - 1))
        secondary_fallback_safe = torch.where(
            effective_gap_mask.gather(2, secondary_fallback.clamp(0, W - 1)),
            secondary_fill_near.clamp(0, W - 1),
            secondary_fallback.clamp(0, W - 1))

        # Resolution priority (best → worst):
        #   1. primary bg (barrier-respected, within max_fill_dist)
        #   2. secondary bg (barrier-respected) — bidirectional bg fallback
        #   3. primary "fixed-step" sample: edge_setback past the gap boundary
        #      in the primary direction, ignoring bg_eligible. Used for gaps
        #      where neither direction found a clean bg pixel — typically
        #      frame-edge gaps. Predictable: just steps N px from the gap edge.
        #   4. secondary "fixed-step" sample (the other direction's fallback)
        # Remaining pixels stay black (no reachable source).
        chosen = secondary_fallback_safe
        has_fill = secondary_fill_has.clone()

        chosen   = torch.where(primary_fill_has,   primary_fallback_safe, chosen)
        has_fill = has_fill | primary_fill_has

        chosen   = torch.where(secondary_bg_valid, secondary_bg_sample, chosen)
        has_fill = has_fill | secondary_bg_valid

        chosen   = torch.where(primary_bg_valid,   primary_bg_sample,   chosen)
        has_fill = has_fill | primary_bg_valid

        fill_idx = chosen.clamp(0, W - 1).unsqueeze(1).expand_as(warped)
        fill_color = warped.gather(3, fill_idx)

        # Reverse-projection: for each gap pixel, compute where the bg pixel
        # that should fill it ORIGINALLY lived in the source image, and sample
        # there directly. Assumes bg depth = 0; inverts the forward warp:
        #   bg_offset_depth = -(convergence_point ** exponent)
        #   bg_pixel_offset = bg_offset_depth * divergence_px + separation_px
        #   src_col         = dest_col - bg_pixel_offset
        # If the source pixel at src_col is itself fg in the original (e.g.,
        # another hair strand happens to occupy that column), we widen the
        # search: find the nearest bg-eligible column in the SOURCE image
        # around src_col. Hair-like scenes have bg visible between strands,
        # so a nearby column almost always carries the correct color.
        # When found, this overrides every horizontal-search strategy above.
        bg_offset_depth  = -(convergence_point ** stereo_offset_exponent)
        bg_pixel_offset  = bg_offset_depth * divergence_px + separation_px
        rp_target_col    = (cols.float() - bg_pixel_offset).round().long()
        rp_target_clamp  = rp_target_col.clamp(0, W - 1)

        # Per-row, per-column nearest bg-eligible column in the SOURCE image
        # (both directions). Computed once per warp, reused for every gap.
        src_bg = normalized_depth < fill_depth_threshold  # [B, H, W]
        src_bg_col_L = torch.where(src_bg, cols, torch.full_like(cols, SENT_L))
        src_bg_col_R = torch.where(src_bg, cols, torch.full_like(cols, SENT_R))
        src_bg_near_L, _ = torch.cummax(src_bg_col_L, dim=2)
        src_bg_near_R    = torch.flip(
            torch.cummin(torch.flip(src_bg_col_R, [2]), dim=2).values, [2])

        # At the target column, look up nearest src-bg to left and right.
        rp_near_L = src_bg_near_L.gather(2, rp_target_clamp)
        rp_near_R = src_bg_near_R.gather(2, rp_target_clamp)
        rp_dist_L = (rp_target_clamp - rp_near_L).clamp(min=0)
        rp_dist_R = (rp_near_R - rp_target_clamp).clamp(min=0)
        rp_has_L  = rp_near_L >= 0
        rp_has_R  = rp_near_R < W

        # Cap the search distance so we don't smear from a clearly-distant bg
        # region in dense-fg scenes.
        RP_MAX_SEARCH = 32  # px around rp_target_col

        # Linearly interpolate between the L and R bg neighbors instead of
        # snap-picking the closer one. Snap-picking caused vertical striping
        # next to fg silhouettes: neighboring rows / neighboring gap pixels
        # would snap to different bg columns whenever rp_target_col landed
        # on a thin fg strand in the source, producing visible discontinuities
        # in the fill. Distance-weighted interpolation produces a smooth fill
        # that varies continuously even when the chosen source column jumps.
        # When only one side has a valid bg neighbor, weight collapses to it.
        rp_only_L = rp_has_L & (~rp_has_R)
        rp_only_R = rp_has_R & (~rp_has_L)
        # Default both-have: weight by inverse distance.
        denom = (rp_dist_L + rp_dist_R).clamp(min=1).float()
        w_L_f = rp_dist_R.float() / denom  # closer L → larger w_L
        w_L_f = torch.where(rp_only_L, torch.ones_like(w_L_f), w_L_f)
        w_L_f = torch.where(rp_only_R, torch.zeros_like(w_L_f), w_L_f)

        rp_chosen_dist = torch.minimum(rp_dist_L.where(rp_has_L, torch.full_like(rp_dist_L, W)),
                                       rp_dist_R.where(rp_has_R, torch.full_like(rp_dist_R, W)))
        rp_in_range    = (rp_target_col >= -RP_MAX_SEARCH) & (rp_target_col < W + RP_MAX_SEARCH)
        rp_valid       = (rp_has_L | rp_has_R) & (rp_chosen_dist <= RP_MAX_SEARCH) & rp_in_range

        # Step edge_setback further into bg in source space, away from the
        # anti-aliased fg silhouette edge. The L neighbor sits just past the
        # fg's left edge in source (the pixel where bg starts); stepping
        # further left lands on cleaner bg. Same idea on the right side.
        # Without this, sampled pixels are fg-tinted from source-image AA
        # and jitter row-to-row with depth-map silhouette noise → striping.
        rp_sample_L_c = (rp_near_L - edge_setback).clamp(0, W - 1)
        rp_sample_R_c = (rp_near_R + edge_setback).clamp(0, W - 1)
        idx_L = rp_sample_L_c.unsqueeze(1).expand_as(image_tensor)
        idx_R = rp_sample_R_c.unsqueeze(1).expand_as(image_tensor)
        color_L = image_tensor.gather(3, idx_L)
        color_R = image_tensor.gather(3, idx_R)
        w_L_4d = w_L_f.unsqueeze(1).expand_as(warped)
        rp_color = color_L * w_L_4d + color_R * (1.0 - w_L_4d)

        # Vertically smooth rp_color by 5 px to suppress residual row-to-row
        # variation (from noisy source depth-edge positions). Only active where
        # the gap region is itself vertically continuous; isolated single-row
        # gap pixels barely shift since the kernel still averages mostly
        # in-gap content. Cheap separable blur, applied uniformly.
        rp_color = F.avg_pool2d(rp_color, kernel_size=(5, 1),
                                stride=1, padding=(2, 0))

        rp_valid_4d = rp_valid.unsqueeze(1).expand_as(warped)
        fill_color  = torch.where(rp_valid_4d, rp_color, fill_color)
        has_fill    = has_fill | rp_valid

        gap_mask_4d = effective_gap_mask.unsqueeze(1).expand_as(warped)
        has_fill_4d = has_fill.unsqueeze(1).expand_as(warped)
        warped = torch.where(gap_mask_4d & has_fill_4d, fill_color, warped)

    return warped, gap_mask, fattened_depth_out


def compute_forward_mask_gpu(depth_tensor, divergence_px, separation_px,
                             stereo_offset_exponent, convergence_point, device):
    """
    Compute pixel-precise gap mask using forward-mapping math, fully vectorized.
    Supports batched inputs [B, H, W].

    Args:
        depth_tensor: [B, H, W] depth map (0-255 or 0-1)
        divergence_px: Horizontal shift in pixels
        separation_px: Additional separation in pixels
        stereo_offset_exponent: Power curve for depth-to-offset
        convergence_point: Depth that maps to zero offset
        device: torch device

    Returns:
        gap_mask: [B, H, W] bool tensor, True = gap (no source pixel lands here)
    """
    B, H, W = depth_tensor.shape

    # Normalize depth to 0-1 per image
    d = depth_tensor.clone()
    d_max_all = d.amax(dim=(1, 2), keepdim=True)
    if (d_max_all > 1.0).any():
        d = d / 255.0
    d_min = d.amin(dim=(1, 2), keepdim=True)
    d_max = d.amax(dim=(1, 2), keepdim=True)
    d_range = d_max - d_min
    normalized_depth = torch.where(
        d_range > 1e-6,
        (d - d_min) / d_range.clamp(min=1e-6),
        torch.zeros_like(d)
    )

    # Apply convergence and exponent
    depth_shifted = normalized_depth - convergence_point
    sign = torch.sign(depth_shifted)
    abs_depth = torch.abs(depth_shifted)
    offset_depth = sign * torch.pow(abs_depth, stereo_offset_exponent)

    # Compute destination column for each source pixel [B, H, W]
    pixel_offset = offset_depth * divergence_px + separation_px
    col_indices = torch.arange(W, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(B, H, W)
    dest_cols = (col_indices + pixel_offset).long()

    # Mark which destinations receive at least one source pixel
    valid = (dest_cols >= 0) & (dest_cols < W)
    dest_clamped = dest_cols.clamp(0, W - 1)

    # scatter_add_ along W dimension (dim=2 for [B, H, W])
    hit_count = torch.zeros(B, H, W, device=device)
    hit_count.scatter_add_(2, dest_clamped, valid.float())

    gap_mask = hit_count < 0.5  # True = no source pixel lands here

    # Dilate gap mask by 1px at depth edges to cover bilinear-blended ghost pixels
    offset_grad = torch.abs(pixel_offset[:, :, 1:] - pixel_offset[:, :, :-1])  # [B, H, W-1]
    depth_edge = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    depth_edge[:, :, :-1] = offset_grad > 1.5
    depth_edge[:, :, 1:] = depth_edge[:, :, 1:] | (offset_grad > 1.5)

    dilated = gap_mask.clone()
    dilated[:, :, 1:] = dilated[:, :, 1:] | (gap_mask[:, :, :-1] & depth_edge[:, :, 1:])
    dilated[:, :, :-1] = dilated[:, :, :-1] | (gap_mask[:, :, 1:] & depth_edge[:, :, :-1])
    gap_mask = dilated

    return gap_mask


def _warp_with_grid(image_tensor, depth_tensor, divergence_px, separation_px,
                    stereo_offset_exponent, convergence_point, device):
    """
    Warp image and return both the result and the warp grid for disocclusion detection.

    Returns:
        warped: [C, H, W] warped image
        valid_mask: [H, W] bool - True where source is in-bounds
        grid: [1, H, W, 2] warp grid
        grid_x_warped: [H, W] warped x coordinates
    """
    C, H, W = image_tensor.shape

    # Normalize depth
    d = depth_tensor.clone()
    if d.max() > 1.0:
        d = d / 255.0
    d_min, d_max = d.min(), d.max()
    if d_max - d_min > 1e-6:
        normalized_depth = (d - d_min) / (d_max - d_min)
    else:
        normalized_depth = torch.zeros_like(d)

    normalized_depth = normalized_depth - convergence_point
    sign = torch.sign(normalized_depth)
    abs_depth = torch.abs(normalized_depth)
    offset_depth = sign * torch.pow(abs_depth, stereo_offset_exponent)

    pixel_offset = offset_depth * divergence_px + separation_px
    offset_normalized = pixel_offset / (W / 2)

    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    grid_x_warped = grid_x - offset_normalized
    grid = torch.stack([grid_x_warped, grid_y], dim=-1).unsqueeze(0)

    warped = F.grid_sample(
        image_tensor.unsqueeze(0), grid,
        mode='bilinear', padding_mode='border', align_corners=True
    ).squeeze(0)

    valid_mask = (grid_x_warped >= -1) & (grid_x_warped <= 1)

    return warped, valid_mask, grid, grid_x_warped


def detect_disocclusions_gpu(depth_tensor, grid, grid_x_warped, device, threshold=0.02):
    """
    Detect disoccluded regions using two complementary signals:

    1. Depth comparison: Warp the depth map with nearest mode and compare.
       If sampled depth >> output depth, the pixel sampled foreground for a
       background position (disoccluded).

    2. Warp gradient: Where the warp field stretches horizontally (derivative > 1),
       pixels are being duplicated/stretched. This catches disocclusions that
       the depth comparison misses due to threshold sensitivity.

    Args:
        depth_tensor: Normalized depth [H, W] in 0-1 range
        grid: Warp grid [1, H, W, 2] from grid_sample
        grid_x_warped: Warped x coordinates [H, W]
        device: torch device
        threshold: Depth difference threshold for disocclusion detection

    Returns:
        disocclusion_mask: [H, W] bool tensor, True = disoccluded pixel
    """
    H, W = depth_tensor.shape

    # Signal 1: Depth comparison
    warped_depth = F.grid_sample(
        depth_tensor.unsqueeze(0).unsqueeze(0), grid,
        mode='nearest', padding_mode='border', align_corners=True
    ).squeeze()  # [H, W]

    depth_diff = warped_depth - depth_tensor
    depth_disoccluded = (depth_diff > threshold)

    # Signal 2: Warp gradient - detect stretched/duplicated regions
    # Compute horizontal derivative of the warp field
    # Where grid_x changes slowly across output columns, multiple outputs sample
    # from the same source region (stretching/duplication)
    warp_grad = torch.zeros_like(grid_x_warped)
    # Forward difference: how much the source x changes per output column
    warp_grad[:, :-1] = torch.abs(grid_x_warped[:, 1:] - grid_x_warped[:, :-1])
    warp_grad[:, -1] = warp_grad[:, -2]

    # Normal warp has gradient ~= pixel_step (2/W for normalized coords)
    # Stretched regions have much larger gradient (source jumps across depth edge)
    pixel_step = 2.0 / W
    stretch_disoccluded = (warp_grad > pixel_step * 3.0)

    # Combine both signals - no dilation, keep pixel-precise like forward mapping
    disoccluded = depth_disoccluded | stretch_disoccluded

    return disoccluded


def interpolate_fill_gpu(image_tensor, mask, device):
    """
    Fill masked regions by stretching border pixels into gaps with linear interpolation.
    Supports batched inputs [B, C, H, W].

    Mimics 'Fill - Naive interpolating': finds the nearest valid pixel on each side
    of a gap and linearly interpolates between them. This gives a natural "stretched
    edge" look without smearing or blurring artifacts.

    Args:
        image_tensor: [B, C, H, W] image tensor
        mask: [B, H, W] bool tensor, True = needs filling
        device: torch device

    Returns:
        filled: [B, C, H, W] tensor with gaps filled
    """
    B, C, H, W = image_tensor.shape
    valid = ~mask  # [B, H, W] True where pixel has real data

    # Column indices [B, H, W]
    cols = torch.arange(W, device=device).unsqueeze(0).unsqueeze(0).expand(B, H, W)

    # --- Left-to-right: find nearest valid pixel to the left ---
    left_valid_col = torch.where(valid, cols, torch.full_like(cols, -1))
    left_nearest_col, _ = torch.cummax(left_valid_col, dim=2)  # along W
    left_dist = (cols - left_nearest_col).float()
    has_left = left_nearest_col >= 0

    # --- Right-to-left: find nearest valid pixel to the right ---
    right_valid_col_flip = torch.where(
        torch.flip(valid, [2]),
        torch.flip(cols, [2]),
        torch.full_like(cols, -1)
    )
    right_nearest_flip, _ = torch.cummax(right_valid_col_flip, dim=2)
    right_nearest_col = torch.flip(right_nearest_flip, [2])
    right_dist = (right_nearest_col - cols).float()
    has_right = right_nearest_col >= 0

    # Gather border colors directly from the nearest valid pixel (no averaging)
    # This gives the "stretching" effect — each border pixel extends into the gap
    left_idx = left_nearest_col.clamp(0, W - 1).unsqueeze(1).expand(B, C, H, W)
    right_idx = right_nearest_col.clamp(0, W - 1).unsqueeze(1).expand(B, C, H, W)
    left_colors = image_tensor.gather(3, left_idx)
    right_colors = image_tensor.gather(3, right_idx)

    # Simple linear interpolation (matches naive_interpolating behavior)
    total_dist = torch.clamp(left_dist + right_dist, min=1.0)
    t = left_dist / total_dist

    # Handle edges with only one valid side
    t = torch.where(~has_left, torch.ones_like(t), t)
    t = torch.where(~has_right, torch.zeros_like(t), t)

    # Expand for broadcasting with [B, C, H, W]
    t_expanded = t.unsqueeze(1)
    interpolated = left_colors * (1.0 - t_expanded) + right_colors * t_expanded
    filled = torch.where(mask.unsqueeze(1), interpolated, image_tensor)

    return filled


def apply_stereo_divergence_gpu_with_fill(image_tensor, depth_tensor, divergence_px, separation_px,
                                           stereo_offset_exponent, convergence_point=0.5, fill_mode='border'):
    """
    GPU-accelerated stereo divergence with configurable fill modes.

    Parameters:
        fill_mode: 'border' (repeat edges), 'zeros' (black fill), 'reflection' (mirror)

    Returns:
        tuple: (warped_image [C,H,W], mask [H,W] indicating valid pixels)
    """
    device = _get_device(depth_tensor, image_tensor)
    image_tensor = image_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    # Handle image format - ensure [C, H, W]
    if image_tensor.dim() == 3:
        if image_tensor.shape[2] in [1, 3, 4]:
            image_tensor = image_tensor.permute(2, 0, 1)

    C, H, W = image_tensor.shape

    # Ensure depth is 2D
    if depth_tensor.dim() == 3:
        depth_tensor = depth_tensor.squeeze()

    # Normalize depth
    depth_min = depth_tensor.min()
    depth_max = depth_tensor.max()
    if depth_max > 1.0:
        depth_tensor = depth_tensor / 255.0
        depth_min = depth_tensor.min()
        depth_max = depth_tensor.max()

    if depth_max - depth_min > 1e-6:
        normalized_depth = (depth_tensor - depth_min) / (depth_max - depth_min)
    else:
        normalized_depth = torch.zeros_like(depth_tensor)

    normalized_depth = normalized_depth - convergence_point

    # Apply exponent with sign preservation
    sign = torch.sign(normalized_depth)
    abs_depth = torch.abs(normalized_depth)
    offset_depth = sign * torch.pow(abs_depth, stereo_offset_exponent)

    pixel_offset = offset_depth * divergence_px + separation_px
    offset_normalized = pixel_offset / (W / 2)

    # Create sampling grid
    y_coords = torch.linspace(-1, 1, H, device=device)
    x_coords = torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

    grid_x_warped = grid_x - offset_normalized
    grid = torch.stack([grid_x_warped, grid_y], dim=-1).unsqueeze(0)

    image_batch = image_tensor.unsqueeze(0)

    # Map fill mode to grid_sample padding mode
    padding_mode_map = {
        'border': 'border',
        'zeros': 'zeros',
        'reflection': 'reflection'
    }
    padding_mode = padding_mode_map.get(fill_mode, 'border')

    warped = F.grid_sample(
        image_batch,
        grid,
        mode='bilinear',
        padding_mode=padding_mode,
        align_corners=True
    )

    # Create mask of valid (in-bounds) pixels
    # A pixel is valid if its source x coordinate is within [-1, 1]
    valid_mask = (grid_x_warped >= -1) & (grid_x_warped <= 1)

    return warped.squeeze(0), valid_mask


def create_stereoimages_gpu(image_tensor, depth_tensor, divergence, separation=0.0, modes=None,
                            stereo_balance=0.0, stereo_offset_exponent=1.0, convergence_point=0.5,
                            skip_flat_depth=False, flat_depth_threshold=1.5,
                            fill_depth_threshold=0.2, edge_setback=4,
                            silhouette_extend=True, extend_radius=8,
                            extend_edge_threshold=10.0, extend_fade_distance=20.0):
    """
    Fully GPU-accelerated stereo image generation with batch support.

    Returns:
        tuple: (list of stereo images [B,C,H,W], left_depth [B,H,W], right_depth [B,H,W], mask [B,H,W])
    """
    if modes is None:
        modes = ['left-right']
    if not isinstance(modes, list):
        modes = [modes]
    if len(modes) == 0:
        return [], None, None, None

    device = _get_device(depth_tensor, image_tensor)
    image_tensor = image_tensor.to(device)
    depth_tensor = depth_tensor.to(device)

    B, _, H, W = image_tensor.shape

    if depth_tensor.amax() <= 1.0:
        depth_tensor = depth_tensor * 255.0

    left_depth = depth_tensor
    right_depth = depth_tensor

    left_divergence = divergence * (1 + stereo_balance)
    right_divergence = divergence * (1 - stereo_balance)

    left_divergence_px = (left_divergence / 100.0) * W
    right_divergence_px = (right_divergence / 100.0) * W
    separation_px = (separation / 100.0) * W

    # Select warp function: ModernGL mesh rasterizer if available, else scatter-based fallback.
    # Mesh path supports gap-fill tuning and silhouette extension; scatter fallback ignores them.
    if MODERNGL_AVAILABLE:
        # Translate user-facing units to internal:
        #   extend_radius          (px, 0 disables) → _FATTEN_RADIUS
        #   extend_edge_threshold  (%, 0-100)       → fraction 0-1 (_FATTEN_JUMP)
        #   extend_fade_distance   (px, 1-50)       → 1/_FATTEN_DECAY
        warp_kwargs = {
            'fill_depth_threshold': fill_depth_threshold,
            'edge_setback': edge_setback,
            'extend_radius': extend_radius if silhouette_extend else 0,
            'extend_edge_threshold': extend_edge_threshold / 100.0,
            'extend_fade_distance': float(extend_fade_distance),
        }
        warp_fn = forward_warp_mesh
    else:
        warp_kwargs = {}
        warp_fn = forward_warp_gpu
    global _warp_path_logged
    if not _warp_path_logged:
        print(f"[ComfyStereo] GPU Warp path: {warp_fn.__name__} "
              f"(MODERNGL_AVAILABLE={MODERNGL_AVAILABLE})")
        _warp_path_logged = True

    # Flat-depth early exit: if every frame in the batch has a depth map that is both
    # gradient-free and produces negligible pixel displacement, skip the expensive warp
    # entirely and duplicate the original image for both eyes.
    # Two conditions must both be true per frame:
    #   1. Max Sobel gradient < _FLAT_GRAD_THRESHOLD (fixed, independent of blur settings).
    #      20 on a 0-255 depth scale ≈ a depth step of ~4-5 gray levels within the
    #      3-pixel Sobel window — conservative enough that real depth structure (cylinder
    #      silhouettes, object edges) still passes, while smooth fades are caught.
    #   2. Effective max displacement < flat_depth_threshold pixels.
    # Only skips if ALL frames in the batch qualify — any non-flat frame keeps the whole batch.
    _FLAT_GRAD_THRESHOLD = 20.0
    skip_warp = False
    if skip_flat_depth:
        max_div_px = max(abs(left_divergence_px), abs(right_divergence_px))
        if max_div_px > 0.01:
            sobel_x_k = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                     dtype=depth_tensor.dtype, device=device).view(1, 1, 3, 3)
            grad_x_flat = F.conv2d(depth_tensor.unsqueeze(1), sobel_x_k, padding=1).squeeze(1)
            max_grad  = grad_x_flat.abs().amax(dim=[1, 2])          # [B]
            d_range   = depth_tensor.amax(dim=[1, 2]) - depth_tensor.amin(dim=[1, 2])  # [B], 0-255
            eff_disp  = (d_range / 255.0) * max_div_px               # [B], pixels
            all_flat  = ((max_grad < _FLAT_GRAD_THRESHOLD) &
                         (eff_disp  < flat_depth_threshold)).all()
            if all_flat:
                skip_warp = True

    left_mask  = torch.zeros(B, H, W, dtype=torch.bool, device=device)
    right_mask = torch.zeros(B, H, W, dtype=torch.bool, device=device)

    if skip_warp:
        left_eye  = image_tensor
        right_eye = image_tensor
    else:
        # OpenGL's top-left fill rule loses 1 px wherever a triangle's *right*
        # edge passes through a pixel center. For the right eye (negative
        # divergence), this loss falls on the disocclusion side of fg and is
        # hidden by gap fill. For the left eye (positive divergence) it falls
        # on the *occlusion* side — producing a visible bg-colored band right
        # at the fg silhouette. Routing the left eye through a flipped warp
        # makes it use the right eye's rasterization path, so the loss lands
        # on the disocclusion side (hidden) for both eyes. Net effect:
        # symmetric artifact profile, no visible bg band at the fg edge.
        if left_divergence < 0.001:
            left_eye = image_tensor
        else:
            image_flipped     = image_tensor.flip(3)
            left_depth_flipped = left_depth.flip(2)
            left_eye_flipped, left_mask_flipped, left_fat_flipped = warp_fn(
                image_flipped, left_depth_flipped, -left_divergence_px, +separation_px,
                stereo_offset_exponent, convergence_point, **warp_kwargs)
            left_eye  = left_eye_flipped.flip(3)
            left_mask = left_mask_flipped.flip(2)
            left_depth = left_fat_flipped.flip(2) * 255.0  # restore 0-255 scale for downstream norm

        if right_divergence < 0.001:
            right_eye = image_tensor
        else:
            right_eye, right_mask, right_fat = warp_fn(
                image_tensor, right_depth, -right_divergence_px, separation_px,
                stereo_offset_exponent, convergence_point, **warp_kwargs)
            right_depth = right_fat * 255.0

    combined_mask = left_mask | right_mask

    # Generate output modes — all dims shift for batch: width=3, height=2
    results = []
    for mode in modes:
        if mode == 'left-right':
            result = torch.cat([left_eye, right_eye], dim=3)
        elif mode == 'right-left':
            result = torch.cat([right_eye, left_eye], dim=3)
        elif mode == 'top-bottom':
            result = torch.cat([left_eye, right_eye], dim=2)
        elif mode == 'bottom-top':
            result = torch.cat([right_eye, left_eye], dim=2)
        elif mode == 'red-cyan-anaglyph':
            result = torch.stack([
                left_eye[:, 0],    # Red from left [B, H, W]
                right_eye[:, 1],   # Green from right
                right_eye[:, 2]    # Blue from right
            ], dim=1)  # -> [B, 3, H, W]
        elif mode == 'left-only':
            result = left_eye
        elif mode == 'only-right':
            result = right_eye
        elif mode == 'cyan-red-reverseanaglyph':
            result = torch.stack([
                right_eye[:, 0],
                left_eye[:, 1],
                left_eye[:, 2]
            ], dim=1)
        else:
            raise ValueError(f'Unknown mode: {mode}')

        results.append(result)

    # Normalize depth maps for output (0-1 range)
    left_depth_out = left_depth / 255.0 if left_depth.amax() > 1.0 else left_depth
    right_depth_out = right_depth / 255.0 if right_depth.amax() > 1.0 else right_depth

    return results, left_depth_out, right_depth_out, combined_mask




def create_stereoimages(original_image, depthmap, divergence, separation=0.0, modes=None,
                        stereo_balance=0.0, stereo_offset_exponent=1.0, fill_technique='polylines_sharp',
                        return_modified_depth=True, convergence_point=0.5,
                        skip_flat_depth=False, flat_depth_threshold=1.5):
    """
    Creates stereoscopic images (CPU path for non-GPU-warp fill techniques).
    Returns (stereo_images, modified_depth) when return_modified_depth, else stereo_images.
    """
    if modes is None:
        modes = ['left-right']
    if not isinstance(modes, list):
        modes = [modes]
    if len(modes) == 0:
        return []

    # Check if inputs are torch tensors (GPU acceleration path)
    use_gpu = isinstance(depthmap, torch.Tensor) and isinstance(original_image, torch.Tensor)

    if use_gpu:
        # GPU-accelerated path - keep tensors on device
        # Ensure grayscale depth map [H, W]
        if depthmap.dim() == 3:
            depthmap = depthmap.squeeze()

        # Normalize depth map to 0-255 range for processing (ComfyUI tensors are 0-1)
        if depthmap.max() <= 1.0:
            depthmap = depthmap * 255.0

        left_depthmap = right_depthmap = depthmap
    else:
        # CPU path - convert to numpy
        original_image = np.asarray(original_image)
        depthmap = np.asarray(depthmap).astype(np.float32)
        left_depthmap = right_depthmap = depthmap

    # Convert to numpy for stereo shift operations (JIT-compiled functions need numpy)
    if use_gpu:
        def _to_np(t):
            return t.cpu().numpy() if t.is_cuda else t.numpy()
        left_depthmap_np = _to_np(left_depthmap)
        right_depthmap_np = _to_np(right_depthmap)
        original_image_np = _to_np(original_image)

        if original_image_np.ndim == 3 and original_image_np.shape[0] == 3:
            original_image_np = original_image_np.transpose(1, 2, 0)
        original_image_np = (np.clip(original_image_np * 255, 0, 255)).astype(np.uint8)

        mod_depth = (_to_np(depthmap) * 255).astype(np.uint8)

        left_depthmap = left_depthmap_np
        right_depthmap = right_depthmap_np
        original_image = original_image_np
    else:
        mod_depth = depthmap.copy()

    # Calculate balanced divergence for each eye
    # When stereo_balance = 0: both eyes get equal divergence (neutral)
    # When stereo_balance > 0: left eye gets more effect (left_divergence increases)
    # When stereo_balance < 0: right eye gets more effect (right_divergence increases)
    left_divergence = divergence * (1 + stereo_balance)
    right_divergence = divergence * (1 - stereo_balance)

    # Flat-depth early exit (CPU path): same two-condition check as the GPU path.
    # Uses the numpy depthmap (0-255 range) and Sobel from scipy.
    _skip_warp = False
    if skip_flat_depth:
        _depth_check = left_depthmap if isinstance(left_depthmap, np.ndarray) else np.asarray(left_depthmap)
        if _depth_check.ndim == 3:
            _depth_check = _depth_check.mean(axis=2)  # collapse channels if present
        _max_div_px = max(abs(left_divergence), abs(right_divergence)) / 100.0 * _depth_check.shape[1]
        if _max_div_px > 0.01:
            _grad = sobel(_depth_check.astype(np.float32), axis=1)
            _max_grad   = np.abs(_grad).max()
            _depth_range = float(_depth_check.max()) - float(_depth_check.min())
            _eff_disp   = (_depth_range / 255.0) * _max_div_px
            if _max_grad < 20.0 and _eff_disp < flat_depth_threshold:
                _skip_warp = True

    if _skip_warp:
        left_eye  = original_image
        right_eye = original_image
    else:
        left_eye = original_image if left_divergence < 0.001 else \
            apply_stereo_divergence(original_image, left_depthmap, +1 * left_divergence, -1 * separation,
                                    stereo_offset_exponent, fill_technique, convergence_point)
        right_eye = original_image if right_divergence < 0.001 else \
            apply_stereo_divergence(original_image, right_depthmap, -1 * right_divergence, separation,
                                    stereo_offset_exponent, fill_technique, convergence_point)
    
    results = []
    for mode in modes:
        if mode == 'left-right':
            results.append(np.hstack([left_eye, right_eye]))
        elif mode == 'right-left':
            results.append(np.hstack([right_eye, left_eye]))
        elif mode == 'top-bottom':
            results.append(np.vstack([left_eye, right_eye]))
        elif mode == 'bottom-top':
            results.append(np.vstack([right_eye, left_eye]))
        elif mode == 'red-cyan-anaglyph':
            results.append(overlap_red_cyan(left_eye, right_eye))
        elif mode == 'left-only':
            results.append(left_eye)
        elif mode == 'only-right':
            results.append(right_eye)
        elif mode == 'cyan-red-reverseanaglyph':
            results.append(overlap_red_cyan(right_eye, left_eye))
        else:
            raise Exception('Unknown mode')
    
    stereo_images = [Image.fromarray(r) for r in results]
    if return_modified_depth:
        mod_img = Image.fromarray(np.clip(mod_depth, 0, 255).astype(np.uint8))
        return stereo_images, mod_img
    return stereo_images

def apply_stereo_divergence(original_image, depth, divergence, separation, stereo_offset_exponent, fill_technique, convergence_point=0.5):
    """
    Dispatches to the desired stereo mapping algorithm.

    Parameters:
        convergence_point: Controls which depth appears at screen plane (0.0-1.0)
                          0.0 = nearest depth at screen (all content recedes)
                          0.5 = mid-depth at screen (balanced, default)
                          1.0 = furthest depth at screen (all content pops out)
    """
    assert original_image.shape[:2] == depth.shape, 'Depthmap and the image must have the same size'
    depth_min = depth.min()
    depth_max = depth.max()

    # Protect against division by zero if depth map is flat (all pixels same value)
    if depth_max == depth_min:
        normalized_depth = np.zeros_like(depth)
    else:
        normalized_depth = (depth - depth_min) / (depth_max - depth_min)

    # Apply convergence point: shift the depth range so convergence_point maps to 0
    # Objects at convergence_point depth will have zero parallax (appear at screen)
    # Objects closer will have positive parallax (pop out)
    # Objects further will have negative parallax (recede)
    normalized_depth = normalized_depth - convergence_point

    divergence_px = (divergence / 100.0) * original_image.shape[1]
    separation_px = (separation / 100.0) * original_image.shape[1]
    
    if fill_technique == 'none_post':
        return apply_stereo_divergence_naive_post(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    if fill_technique == 'inverse_post':
        return apply_stereo_divergence_inverse_post(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    if fill_technique == 'hybrid_edge_plus':
        return apply_stereo_divergence_hybrid_edge_plus(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    if fill_technique == 'hybrid_edge':
        return apply_stereo_divergence_hybrid_edge(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    if fill_technique in ['none', 'naive', 'naive_interpolating']:
        return apply_stereo_divergence_naive(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, fill_technique)
    if fill_technique in ['polylines_soft', 'polylines_sharp']:
        return apply_stereo_divergence_polylines(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, fill_technique)
    if fill_technique == 'inverse':
        return apply_stereo_divergence_inverse(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)

    return original_image  # Fallback

@njit(parallel=True)
def enhanced_inverse_mapping_with_mask(original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float):
    """
    Enhanced inverse mapping that distributes each source pixel's color over three adjacent columns
    using a Gaussian kernel. Returns both the accumulated image and a binary mask.
    """
    h, w, c = original_image.shape
    accum = np.zeros((h, w, c), dtype=np.float32)
    weight_sum = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.uint8)
    sigma = 1.0  # standard deviation for the subpixel kernel
    for row in prange(h):
        for x in range(w):
            d = normalized_depth[row, x]
            sign_d = 1.0 if d >= 0.0 else -1.0
            offset = sign_d * (abs(d) ** stereo_offset_exponent) * divergence_px
            dest_x = x + 0.5 + offset + separation_px
            j_center = int(math.floor(dest_x))
            for d in (-1, 0, 1):
                j = j_center + d
                if j >= 0 and j < w:
                    diff = dest_x - j
                    wght = math.exp(- (diff * diff) / (2 * sigma * sigma))
                    for ch in range(c):
                        accum[row, j, ch] += original_image[row, x, ch] * wght
                    weight_sum[row, j] += wght
                    mask[row, j] = 1
    # Normalize the accumulated image.
    output = np.zeros((h, w, c), dtype=np.uint8)
    for row in range(h):
        for j in range(w):
            if weight_sum[row, j] > 0:
                for ch in range(c):
                    val = accum[row, j, ch] / weight_sum[row, j]
                    if val < 0:
                        val = 0
                    elif val > 255:
                        val = 255
                    output[row, j, ch] = int(val)
    return output, mask


@njit(parallel=True)
def naive_mapping_with_mask(original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float):
    h, w, c = original_image.shape
    derived_image = np.zeros_like(original_image)
    filled = np.zeros(h * w, dtype=np.uint8)
    for row in prange(h):
        if divergence_px < 0:
            rng = range(w)
        else:
            rng = range(w - 1, -1, -1)
        for col in rng:
            d = normalized_depth[row, col]
            sign_d = 1.0 if d >= 0.0 else -1.0
            offset = sign_d * (abs(d) ** stereo_offset_exponent) * divergence_px + separation_px
            col_d = col + int(offset)
            if 0 <= col_d < w:
                derived_image[row, col_d] = original_image[row, col]
                filled[row * w + col_d] = 1
    filled_mask = np.empty((h, w), dtype=np.uint8)
    for i in range(h * w):
        filled_mask.flat[i] = filled[i]
    return derived_image, filled_mask


@njit(parallel=True)
def inverse_mapping_with_mask(original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float):
    h, w, c = original_image.shape
    derived_image = np.zeros_like(original_image)
    mask = np.zeros((h, w), dtype=np.uint8)
    for row in prange(h):
        depth_buffer = np.full(w, -1.0, dtype=np.float32)
        for x in range(w):
            d = normalized_depth[row, x]
            sign_d = 1.0 if d >= 0.0 else -1.0
            offset = sign_d * (abs(d) ** stereo_offset_exponent) * divergence_px
            dest_x = x + 0.5 + offset + separation_px
            closeness = normalized_depth[row, x]
            j = int(np.floor(dest_x))
            frac = dest_x - j
            if 0 <= j < w:
                if closeness > depth_buffer[j]:
                    derived_image[row, j] = original_image[row, x]
                    depth_buffer[j] = closeness
                    mask[row, j] = 1
            if 0 <= j + 1 < w:
                if closeness > depth_buffer[j + 1]:
                    derived_image[row, j + 1] = original_image[row, x]
                    depth_buffer[j + 1] = closeness
                    mask[row, j + 1] = 1
    return derived_image, mask

@njit(parallel=True)
def apply_stereo_divergence_inverse(original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float):
    h, w, c = original_image.shape
    derived_image = np.zeros_like(original_image)
    for row in prange(h):
        depth_buffer = np.full(w, -1.0, dtype=np.float32)
        for x in range(w):
            d = normalized_depth[row, x]
            sign_d = 1.0 if d >= 0.0 else -1.0
            offset = sign_d * (abs(d) ** stereo_offset_exponent) * divergence_px
            dest_x = x + 0.5 + offset + separation_px
            closeness = normalized_depth[row, x]
            j = int(np.floor(dest_x))
            frac = dest_x - j
            if 0 <= j < w:
                if closeness > depth_buffer[j]:
                    derived_image[row, j] = original_image[row, x]
                    depth_buffer[j] = closeness
            if 0 <= j + 1 < w:
                if closeness > depth_buffer[j + 1]:
                    derived_image[row, j + 1] = original_image[row, x]
                    depth_buffer[j + 1] = closeness
    return derived_image


def rgb2gray(image):
    """Convert an RGB image (H x W x 3) to grayscale using standard weights."""
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def edge_aware_gap_fill(image, mask, guidance, window_size=3, sigma_s=1.0, sigma_r=10.0):
    """
    For each pixel not filled (mask==0) in 'image', perform 2D interpolation using neighboring
    pixels that are filled. The weights are computed based on both spatial distance and guidance difference.
    'guidance' is a single-channel (grayscale) image used to preserve edges.
    """
    h, w, c = image.shape
    filled = image.astype(np.float32).copy()
    half_win = window_size // 2
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0:
                new_val = np.zeros(c, dtype=np.float32)
                weight_total = 0.0
                for di in range(-half_win, half_win+1):
                    for dj in range(-half_win, half_win+1):
                        ni = i + di
                        nj = j + dj
                        if ni >= 0 and ni < h and nj >= 0 and nj < w:
                            if mask[ni, nj] != 0:
                                dsq = di*di + dj*dj
                                w_s = math.exp(- dsq / (2 * sigma_s * sigma_s))
                                diff = guidance[i, j] - guidance[ni, nj]
                                w_r = math.exp(- (diff*diff) / (2 * sigma_r * sigma_r))
                                wght = w_s * w_r
                                new_val += image[ni, nj].astype(np.float32) * wght
                                weight_total += wght
                if weight_total > 0:
                    filled[i, j] = new_val / weight_total
    return np.clip(filled, 0, 255).astype(np.uint8)



def apply_stereo_divergence_hybrid_edge_plus(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent):
    """
    Hybrid method that first uses enhanced inverse mapping with 2D edge-aware gap filling
    (the "hybrid_edge" method) and then, for any pixels that remain unfilled (detected as black),
    uses a fallback from the polylines_soft method.
    """
    # First, get the initial result using our enhanced method.
    base_img, mask = enhanced_inverse_mapping_with_mask(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    # Compute a guidance image from the original (grayscale)
    guidance = rgb2gray(np.asarray(original_image))
    # Apply 2D edge-aware gap filling:
    filled_img = edge_aware_gap_fill(base_img, mask, guidance, window_size=3, sigma_s=1.0, sigma_r=10.0)
    
    # Next, compute an alternative mapping using the polylines_soft method.
    poly_img = apply_stereo_divergence_polylines(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, 'polylines_soft')
    
    # Finally, combine the two: for any pixel where filled_img remains black, use the poly_img pixel.
    h, w, c = filled_img.shape
    final_img = filled_img.copy()
    for i in range(h):
        for j in range(w):
            # If a pixel is unfilled, we assume its channels are all zero.
            if (final_img[i, j, 0] == 0 and final_img[i, j, 1] == 0 and final_img[i, j, 2] == 0):
                final_img[i, j] = poly_img[i, j]
    return final_img

def apply_stereo_divergence_naive_post(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent):
    base_img, mask = naive_mapping_with_mask(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    h, w, c = base_img.shape
    output = base_img.astype(np.float32).copy()
    for row in range(h):
        x_coords = np.arange(w, dtype=np.float32)
        valid = np.nonzero(mask[row])[0]
        if valid.size == 0:
            continue
        for ch in range(c):
            row_data = base_img[row, :, ch].astype(np.float32)
            interpolated = np.interp(x_coords, valid.astype(np.float32), row_data[valid])
            output[row, :, ch] = interpolated
    return output.astype(np.uint8)


def apply_stereo_divergence_inverse_post(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent):
    base_img, mask = inverse_mapping_with_mask(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    h, w, c = base_img.shape
    output = base_img.astype(np.float32).copy()
    for row in range(h):
        x_coords = np.arange(w, dtype=np.float32)
        valid = np.nonzero(mask[row])[0]
        if valid.size == 0:
            continue
        for ch in range(c):
            row_data = base_img[row, :, ch].astype(np.float32)
            interpolated = np.interp(x_coords, valid.astype(np.float32), row_data[valid])
            output[row, :, ch] = interpolated
    return output.astype(np.uint8)



def apply_stereo_divergence_hybrid_edge(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent):
    """
    Hybrid method: uses enhanced inverse mapping (with subpixel distribution over 3 columns) to produce
    an initial stereo image and mask, then applies 2D, edge-aware gap filling.
    """
    # First, get the base image and mask using the enhanced mapping.
    base_img, mask = enhanced_inverse_mapping_with_mask(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    # Compute a guidance image (grayscale) from the original image.
    guidance = rgb2gray(np.asarray(original_image))
    # Apply 2D edge-aware gap filling over the entire image.
    filled_img = edge_aware_gap_fill(base_img, mask, guidance, window_size=3, sigma_s=1.0, sigma_r=10.0)
    return filled_img

@njit(parallel=True)
def apply_stereo_divergence_naive(
        original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float,
        fill_technique: str):
    h, w, c = original_image.shape

    derived_image = np.zeros_like(original_image)
    filled = np.zeros(h * w, dtype=np.uint8)

    for row in prange(h):
        # Swipe order should ensure that pixels that are closer overwrite
        # (at their destination) pixels that are less close
        for col in range(w) if divergence_px < 0 else range(w - 1, -1, -1):
            d = normalized_depth[row][col]
            sign_d = 1.0 if d >= 0.0 else -1.0
            col_d = col + int(sign_d * (abs(d) ** stereo_offset_exponent) * divergence_px + separation_px)
            if 0 <= col_d < w:
                derived_image[row][col_d] = original_image[row][col]
                filled[row * w + col_d] = 1

    # Fill the gaps
    if fill_technique == 'naive_interpolating':
        for row in range(h):
            for l_pointer in range(w):
                if sum(derived_image[row][l_pointer]) != 0 or filled[row * w + l_pointer]:
                    continue
                l_border = derived_image[row][l_pointer - 1] if l_pointer > 0 else np.zeros(3, dtype=np.uint8)
                r_border = np.zeros(3, dtype=np.uint8)
                r_pointer = l_pointer + 1
                while r_pointer < w:
                    if sum(derived_image[row][r_pointer]) != 0 and filled[row * w + r_pointer]:
                        r_border = derived_image[row][r_pointer]
                        break
                    r_pointer += 1
                if sum(l_border) == 0:
                    l_border = r_border
                elif sum(r_border) == 0:
                    r_border = l_border
                total_steps = 1 + r_pointer - l_pointer
                step = (r_border.astype(np.float32) - l_border) / total_steps
                for col in range(l_pointer, r_pointer):
                    derived_image[row][col] = l_border + (step * (col - l_pointer + 1)).astype(np.uint8)
        return derived_image
    elif fill_technique == 'naive':
        derived_fix = np.copy(derived_image)
        for pos in np.where(filled == 0)[0]:
            row = pos // w
            col = pos % w
            row_times_w = row * w
            for offset in range(1, abs(int(divergence_px)) + 2):
                r_offset = col + offset
                l_offset = col - offset
                if r_offset < w and filled[row_times_w + r_offset]:
                    derived_fix[row][col] = derived_image[row][r_offset]
                    break
                if 0 <= l_offset and filled[row_times_w + l_offset]:
                    derived_fix[row][col] = derived_image[row][l_offset]
                    break
        return derived_fix
    else:  # none
        return derived_image

@njit(parallel=True)
def apply_stereo_divergence_polylines(original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float, fill_technique: str):
    EPSILON = 1e-7
    PIXEL_HALF_WIDTH = 0.45 if fill_technique == 'polylines_sharp' else 0.0
    h, w, c = original_image.shape
    derived_image = np.zeros_like(original_image)
    for row in prange(h):
        pt = np.zeros((5 + 2 * w, 3), dtype=np.float32)
        pt_end: int = 0
        pt[pt_end] = [-1.0 * w, 0.0, 0.0]
        pt_end += 1
        for col in range(0, w):
            d = normalized_depth[row][col]
            sign_d = 1.0 if d >= 0.0 else -1.0
            coord_d = sign_d * (abs(d) ** stereo_offset_exponent) * divergence_px
            coord_x = col + 0.5 + coord_d + separation_px
            if PIXEL_HALF_WIDTH < EPSILON:
                pt[pt_end] = [coord_x, abs(coord_d), col]
                pt_end += 1
            else:
                pt[pt_end] = [coord_x - PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt[pt_end + 1] = [coord_x + PIXEL_HALF_WIDTH, abs(coord_d), col]
                pt_end += 2
        pt[pt_end] = [2.0 * w, 0.0, w - 1]
        pt_end += 1
        sg_end: int = pt_end - 1
        sg = np.zeros((sg_end, 6), dtype=np.float32)
        for i in range(sg_end):
            sg[i] += np.concatenate((pt[i], pt[i + 1]))
        for i in range(1, sg_end):
            u = i - 1
            while pt[u][0] > pt[u + 1][0] and 0 <= u:
                pt[u], pt[u + 1] = np.copy(pt[u + 1]), np.copy(pt[u])
                sg[u], sg[u + 1] = np.copy(sg[u + 1]), np.copy(sg[u])
                u -= 1
        csg = np.zeros((5 * int(abs(divergence_px)) + 25, 6), dtype=np.float32)
        csg_end: int = 0
        sg_pointer: int = 0
        pt_i: int = 0
        for col in range(w):
            color = np.full(c, 0.5, dtype=np.float32)
            while pt[pt_i][0] < col:
                pt_i += 1
            pt_i -= 1
            while pt[pt_i][0] < col + 1:
                coord_from = max(col, pt[pt_i][0]) + EPSILON
                coord_to = min(col + 1, pt[pt_i + 1][0]) - EPSILON
                significance = coord_to - coord_from
                coord_center = coord_from + 0.5 * significance
                while sg_pointer < sg_end and sg[sg_pointer][0] < coord_center:
                    csg[csg_end] = sg[sg_pointer]
                    sg_pointer += 1
                    csg_end += 1
                csg_i = 0
                while csg_i < csg_end:
                    if csg[csg_i][3] < coord_center:
                        csg[csg_i] = csg[csg_end - 1]
                        csg_end -= 1
                    else:
                        csg_i += 1
                best_csg_i: int = 0
                if csg_end != 1:
                    best_csg_closeness: float = -EPSILON
                    for csg_i in range(csg_end):
                        ip_k = (coord_center - csg[csg_i][0]) / (csg[csg_i][3] - csg[csg_i][0])
                        closeness = (1.0 - ip_k) * csg[csg_i][1] + ip_k * csg[csg_i][4]
                        if best_csg_closeness < closeness and 0.0 < ip_k < 1.0:
                            best_csg_closeness = closeness
                            best_csg_i = csg_i
                col_l: int = int(csg[best_csg_i][2] + EPSILON)
                col_r: int = int(csg[best_csg_i][5] + EPSILON)
                if col_l == col_r:
                    color += original_image[row][col_l] * significance
                else:
                    ip_k = (coord_center - csg[best_csg_i][0]) / (csg[best_csg_i][3] - csg[best_csg_i][0])
                    color += (original_image[row][col_l] * (1.0 - ip_k) +
                              original_image[row][col_r] * ip_k
                              ) * significance
                pt_i += 1
            derived_image[row][col] = np.asarray(color, dtype=np.uint8)
    return derived_image



@njit(parallel=True)
def overlap_red_cyan(im1, im2):
    width1 = im1.shape[1]
    height1 = im1.shape[0]
    width2 = im2.shape[1]
    height2 = im2.shape[0]
    composite = np.zeros((height2, width2, 3), np.uint8)
    for i in prange(height1):
        for j in range(width1):
            composite[i, j, 0] = im1[i, j, 0]
    for i in prange(height2):
        for j in range(width2):
            composite[i, j, 1] = im2[i, j, 1]
            composite[i, j, 2] = im2[i, j, 2]
    return composite
