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

def blur_depth_map(depth, sigma):
    """
    Applies a separable Gaussian blur to a 2D depth map.
    'sigma' is the standard deviation in pixels.
    If sigma <= 0, the original depth map is returned.
    """
    if sigma <= 0:
        return depth
    radius = int(3 * sigma)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(- (x ** 2) / (2 * sigma * sigma))
    kernel /= kernel.sum()
    
    h, w = depth.shape
    # Horizontal pass
    blurred = np.empty_like(depth, dtype=np.float64)
    for i in range(h):
        row = depth[i, :]
        padded = np.pad(row, (radius, radius), mode='edge')
        conv = np.convolve(padded, kernel, mode='valid')
        blurred[i, :] = conv
    # Vertical pass
    final = np.empty_like(blurred, dtype=np.float64)
    for j in range(w):
        col = blurred[:, j]
        padded = np.pad(col, (radius, radius), mode='edge')
        conv = np.convolve(padded, kernel, mode='valid')
        final[:, j] = conv
    return final

def edge_selective_blur_depth_map(depth, sigma, edge_threshold):
    """
    Applies a global edge-selective blur (ignoring direction) to the depth map.
    (This is our previous implementation.)
    """
    h, w = depth.shape
    # Compute gradients using Sobel operators.
    padded = np.pad(depth, 1, mode='edge')
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]])
    grad_x = np.zeros_like(depth)
    grad_y = np.zeros_like(depth)
    for i in range(h):
        for j in range(w):
            region = padded[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(region * sobel_x)
            grad_y[i, j] = np.sum(region * sobel_y)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    # Create weight: values >= edge_threshold get full weight.
    weight = np.minimum(grad_mag / edge_threshold, 1.0)
    blurred = blur_depth_map(depth, sigma)
    output = (1.0 - weight) * depth + weight * blurred
    return output

def left_direction_aware_blur_depth_map(depth, sigma, edge_threshold):
    """
    For the left image: only blur pixels where the horizontal gradient is positive
    (i.e. a dark->light transition). The weight is given by:
         weight = min( (grad / edge_threshold), 1 )   if grad > 0, else 0.
    """
    h, w = depth.shape
    grad = np.zeros_like(depth)
    # Compute a simple horizontal gradient using central differences.
    for i in range(h):
        row = depth[i, :]
        padded = np.pad(row, (1, 1), mode='edge')
        for j in range(w):
            grad[i, j] = (padded[j+2] - padded[j]) / 2.0
    weight = np.where(grad > 0, np.minimum(grad / edge_threshold, 1.0), 0.0)
    blurred = blur_depth_map(depth, sigma)
    return (1.0 - weight) * depth + weight * blurred

def right_direction_aware_blur_depth_map(depth, sigma, edge_threshold):
    """
    For the right image: only blur pixels where the horizontal gradient is negative
    (i.e. a light->dark transition). The weight is given by:
         weight = min( (|grad| / edge_threshold), 1 )   if grad < 0, else 0.
    """
    h, w = depth.shape
    grad = np.zeros_like(depth)
    for i in range(h):
        row = depth[i, :]
        padded = np.pad(row, (1, 1), mode='edge')
        for j in range(w):
            grad[i, j] = (padded[j+2] - padded[j]) / 2.0
    weight = np.where(grad < 0, np.minimum(np.abs(grad) / edge_threshold, 1.0), 0.0)
    blurred = blur_depth_map(depth, sigma)
    return (1.0 - weight) * depth + weight * blurred

def create_stereoimages(original_image, depthmap, divergence, separation=0.0, modes=None,
                        stereo_balance=0.0, stereo_offset_exponent=1.0, fill_technique='polylines_sharp',
                        depth_blur_sigma=0.0, depth_blur_edge_threshold=10.0,
                        direction_aware_depth_blur=False, return_modified_depth=True):
                            
    """
    Creates stereoscopic images.
    
    Parameters:
      - original_image: The source image.
      - depthmap: A depth map (white = near, black = far).
      - divergence: 3D effect strength in percentages.
      - separation: Additional horizontal shift percentage.
      - modes: List of output modes (e.g., 'left-right', 'red-cyan-anaglyph').
      - stereo_balance: How divergence is split between the two eyes.
      - stereo_offset_exponent: Exponent controlling the depth-to-offset mapping.
      - fill_technique: Method used to fill gaps (e.g., 'polylines_sharp', etc.).
      - depth_blur_sigma: Standard deviation for Gaussian blur.
      - depth_blur_edge_threshold: Gradient threshold controlling blur strength.
      - direction_aware_depth_blur: If True, generate two depth maps:
            one for the left eye (blurring only on positive horizontal gradients)
            and one for the right eye (blurring only on negative horizontal gradients).
      - return_modified_depth: If True, return the modified depth maps as well.
    
    Returns:
      If return_modified_depth is True and direction_aware_depth_blur is True, a tuple
      (stereo_images, left_modified_depth, right_modified_depth) is returned. Otherwise,
      if direction_aware_depth_blur is False and return_modified_depth is True, a tuple
      (stereo_images, modified_depth) is returned, where modified_depth is produced by the
      global edge-selective blur. If return_modified_depth is False, only stereo_images is returned.
    """
    if depth_blur_sigma > 0:
        direction_aware_depth_blur = True
    
    if modes is None:
        modes = ['left-right']
    if not isinstance(modes, list):
        modes = [modes]
    if len(modes) == 0:
        return []
    
    original_image = np.asarray(original_image)
    depthmap = np.asarray(depthmap).astype(np.float64)
    
    if direction_aware_depth_blur:
        left_depthmap = left_direction_aware_blur_depth_map(depthmap, depth_blur_sigma, depth_blur_edge_threshold)
        right_depthmap = right_direction_aware_blur_depth_map(depthmap, depth_blur_sigma, depth_blur_edge_threshold)
    else:
        # Global edge-selective blur (as before)
        if depth_blur_sigma > 0:
            depthmap = edge_selective_blur_depth_map(depthmap, depth_blur_sigma, depth_blur_edge_threshold)
        left_depthmap = right_depthmap = depthmap

    # Save modified depth maps for output.
    if direction_aware_depth_blur:
        mod_left = left_depthmap.copy()
        mod_right = right_depthmap.copy()
    else:
        mod_depth = depthmap.copy()
    
    balance = (stereo_balance + 1) / 2
    left_eye = original_image if balance < 0.001 else \
        apply_stereo_divergence(original_image, left_depthmap, +1 * divergence, -1 * separation,
                                stereo_offset_exponent, fill_technique)
    right_eye = original_image if balance > 0.999 else \
        apply_stereo_divergence(original_image, right_depthmap, -1 * divergence, separation,
                                stereo_offset_exponent, fill_technique)
    
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
        if direction_aware_depth_blur:
            left_mod_img = Image.fromarray(np.clip(mod_left, 0, 255).astype(np.uint8))
            right_mod_img = Image.fromarray(np.clip(mod_right, 0, 255).astype(np.uint8))
            return stereo_images, left_mod_img, right_mod_img
        else:
            mod_img = Image.fromarray(np.clip(mod_depth, 0, 255).astype(np.uint8))
            return stereo_images, mod_img
    else:
        return stereo_images

def apply_stereo_divergence(original_image, depth, divergence, separation, stereo_offset_exponent, fill_technique):
    """
    Dispatches to the desired stereo mapping algorithm.
    """
    assert original_image.shape[:2] == depth.shape, 'Depthmap and the image must have the same size'
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
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
    accum = np.zeros((h, w, c), dtype=np.float64)
    weight_sum = np.zeros((h, w), dtype=np.float64)
    mask = np.zeros((h, w), dtype=np.uint8)
    sigma = 1.0  # standard deviation for the subpixel kernel
    for row in prange(h):
        for x in range(w):
            offset = (normalized_depth[row, x] ** stereo_offset_exponent) * divergence_px
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
            offset = (normalized_depth[row, col] ** stereo_offset_exponent) * divergence_px + separation_px
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
        depth_buffer = np.full(w, -1.0, dtype=np.float64)
        for x in range(w):
            offset = (normalized_depth[row, x] ** stereo_offset_exponent) * divergence_px
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
        depth_buffer = np.full(w, -1.0, dtype=np.float64)
        for x in range(w):
            offset = (normalized_depth[row, x] ** stereo_offset_exponent) * divergence_px
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
    filled = image.astype(np.float64).copy()
    half_win = window_size // 2
    for i in range(h):
        for j in range(w):
            if mask[i, j] == 0:
                new_val = np.zeros(c, dtype=np.float64)
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
                                new_val += image[ni, nj].astype(np.float64) * wght
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
    output = base_img.astype(np.float64).copy()
    for row in range(h):
        x_coords = np.arange(w, dtype=np.float64)
        valid = np.nonzero(mask[row])[0]
        if valid.size == 0:
            continue
        for ch in range(c):
            row_data = base_img[row, :, ch].astype(np.float64)
            interpolated = np.interp(x_coords, valid.astype(np.float64), row_data[valid])
            output[row, :, ch] = interpolated
    return output.astype(np.uint8)


def apply_stereo_divergence_inverse_post(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent):
    base_img, mask = inverse_mapping_with_mask(original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent)
    h, w, c = base_img.shape
    output = base_img.astype(np.float64).copy()
    for row in range(h):
        x_coords = np.arange(w, dtype=np.float64)
        valid = np.nonzero(mask[row])[0]
        if valid.size == 0:
            continue
        for ch in range(c):
            row_data = base_img[row, :, ch].astype(np.float64)
            interpolated = np.interp(x_coords, valid.astype(np.float64), row_data[valid])
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
            col_d = col + int((normalized_depth[row][col] ** stereo_offset_exponent) * divergence_px + separation_px)
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
                step = (r_border.astype(np.float_) - l_border) / total_steps
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
        pt = np.zeros((5 + 2 * w, 3), dtype=np.float_)
        pt_end: int = 0
        pt[pt_end] = [-1.0 * w, 0.0, 0.0]
        pt_end += 1
        for col in range(0, w):
            coord_d = (normalized_depth[row][col] ** stereo_offset_exponent) * divergence_px
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
        sg = np.zeros((sg_end, 6), dtype=np.float_)
        for i in range(sg_end):
            sg[i] += np.concatenate((pt[i], pt[i + 1]))
        for i in range(1, sg_end):
            u = i - 1
            while pt[u][0] > pt[u + 1][0] and 0 <= u:
                pt[u], pt[u + 1] = np.copy(pt[u + 1]), np.copy(pt[u])
                sg[u], sg[u + 1] = np.copy(sg[u + 1]), np.copy(sg[u])
                u -= 1
        csg = np.zeros((5 * int(abs(divergence_px)) + 25, 6), dtype=np.float_)
        csg_end: int = 0
        sg_pointer: int = 0
        pt_i: int = 0
        for col in range(w):
            color = np.full(c, 0.5, dtype=np.float_)
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
