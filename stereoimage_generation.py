try:
    from numba import njit, prange
except Exception as e:
    print(f"WARINING! Numba failed to import! Stereoimage generation will be much slower! ({str(e)})")
    from builtins import range as prange
    def njit(parallel=False):
        def Inner(func): return lambda *args, **kwargs: func(*args, **kwargs)
        return Inner
import numpy as np
from PIL import Image
import math

# --- Depth Map Blurring Functions ---
def blur_depth_map(depth, sigma):
    """
    Applies a separable Gaussian blur to a 2D depth map.
    If sigma <= 0, returns the original depth map.
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
    Applies a global edge-selective blur to the depth map.
    The gradient is computed via Sobel operators.
    """
    h, w = depth.shape
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
    weight = np.minimum(grad_mag / edge_threshold, 1.0)
    blurred = blur_depth_map(depth, sigma)
    output = (1.0 - weight) * depth + weight * blurred
    return output

def left_direction_aware_blur_depth_map(depth, sigma, edge_threshold):
    """
    For the left image: blurs only pixels where the horizontal gradient is positive.
    Weight is proportional to the gradient (capped by edge_threshold).
    """
    h, w = depth.shape
    grad = np.zeros_like(depth)
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
    For the right image: blurs only pixels where the horizontal gradient is negative.
    Weight is proportional to the absolute gradient (capped by edge_threshold).
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

# --- Stereoimage Generation Functions ---
def create_stereoimages(original_image, depthmap, divergence, separation=0.0, modes=None,
                        stereo_balance=0.0, stereo_offset_exponent=1.0, fill_technique='polylines_sharp',
                        depth_blur_sigma=0.0, depth_blur_edge_threshold=10.0, direction_aware_depth_blur=True):
    """
    Creates stereoscopic images.
    
    Extended parameters for depth blurring:
      - depth_blur_sigma: Standard deviation for the Gaussian blur.
      - depth_blur_edge_threshold: Threshold for edge-selective blurring.
      - direction_aware_depth_blur: When True, applies separate directional blurs for left/right images.
    """
    if modes is None:
        modes = ['left-right']
    if not isinstance(modes, list):
        modes = [modes]
    if len(modes) == 0:
        return []
    
    original_image = np.asarray(original_image)
    depthmap = np.asarray(depthmap).astype(np.float64)
    
    # Apply depth map blurring.
    if direction_aware_depth_blur:
        left_depthmap = left_direction_aware_blur_depth_map(depthmap, depth_blur_sigma, depth_blur_edge_threshold)
        right_depthmap = right_direction_aware_blur_depth_map(depthmap, depth_blur_sigma, depth_blur_edge_threshold)
    else:
        if depth_blur_sigma > 0:
            depthmap = edge_selective_blur_depth_map(depthmap, depth_blur_sigma, depth_blur_edge_threshold)
        left_depthmap = right_depthmap = depthmap
    
    balance = (stereo_balance + 1) / 2
    left_eye = original_image if balance < 0.001 else \
        apply_stereo_divergence(original_image, left_depthmap, +1 * divergence * balance, -1 * separation,
                                stereo_offset_exponent, fill_technique)
    right_eye = original_image if balance > 0.999 else \
        apply_stereo_divergence(original_image, right_depthmap, -1 * divergence * (1 - balance), separation,
                                stereo_offset_exponent, fill_technique)

    results = []
    for mode in modes:
        if mode == 'left-right':  # Most popular format. Common use case: displaying in HMD.
            results.append(np.hstack([left_eye, right_eye]))
        elif mode == 'right-left':  # Cross-viewing
            results.append(np.hstack([right_eye, left_eye]))
        elif mode == 'top-bottom':
            results.append(np.vstack([left_eye, right_eye]))
        elif mode == 'bottom-top':
            results.append(np.vstack([right_eye, left_eye]))
        elif mode == 'red-cyan-anaglyph':  # Anaglyph for red-cyan glasses
            results.append(overlap_red_cyan(left_eye, right_eye))
        elif mode == 'left-only':
            results.append(left_eye)
        elif mode == 'only-right':
            results.append(right_eye)
        elif mode == 'cyan-red-reverseanaglyph':  # Alternate anaglyph style
            results.append(overlap_red_cyan(right_eye, left_eye))
        else:
            raise Exception('Unknown mode')
    return [Image.fromarray(r) for r in results]

def apply_stereo_divergence(original_image, depth, divergence, separation, stereo_offset_exponent, fill_technique):
    """
    Dispatches to the desired stereo mapping algorithm.
    Assumes that original_image and depth have matching dimensions.
    """
    assert original_image.shape[:2] == depth.shape, 'Depthmap and the image must have the same size'
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = (depth - depth_min) / (depth_max - depth_min)
    divergence_px = (divergence / 100.0) * original_image.shape[1]
    separation_px = (separation / 100.0) * original_image.shape[1]

    if fill_technique in ['none', 'naive', 'naive_interpolating']:
        return apply_stereo_divergence_naive(
            original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, fill_technique
        )
    if fill_technique in ['polylines_soft', 'polylines_sharp']:
        return apply_stereo_divergence_polylines(
            original_image, normalized_depth, divergence_px, separation_px, stereo_offset_exponent, fill_technique
        )

@njit(parallel=False)
def apply_stereo_divergence_naive(
        original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float,
        fill_technique: str):
    h, w, c = original_image.shape

    derived_image = np.zeros_like(original_image)
    filled = np.zeros(h * w, dtype=np.uint8)

    for row in prange(h):
        # Swipe order ensures that closer pixels overwrite farther ones.
        for col in range(w) if divergence_px < 0 else range(w - 1, -1, -1):
            col_d = col + int((normalized_depth[row][col] ** stereo_offset_exponent) * divergence_px + separation_px)
            if 0 <= col_d < w:
                derived_image[row][col_d] = original_image[row][col]
                filled[row * w + col_d] = 1

    # Fill the gaps using interpolation if requested.
    if fill_technique == 'naive_interpolating':
        for row in range(h):
            for l_pointer in range(w):
                if np.sum(derived_image[row][l_pointer]) != 0 or filled[row * w + l_pointer]:
                    continue
                l_border = derived_image[row][l_pointer - 1] if l_pointer > 0 else np.zeros(3, dtype=np.uint8)
                r_border = np.zeros(3, dtype=np.uint8)
                r_pointer = l_pointer + 1
                while r_pointer < w:
                    if np.sum(derived_image[row][r_pointer]) != 0 and filled[row * w + r_pointer]:
                        r_border = derived_image[row][r_pointer]
                        break
                    r_pointer += 1
                if np.sum(l_border) == 0:
                    l_border = r_border
                elif np.sum(r_border) == 0:
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
    else:  # fill_technique == 'none'
        return derived_image

@njit(parallel=True)
def apply_stereo_divergence_polylines(
        original_image, normalized_depth, divergence_px: float, separation_px: float, stereo_offset_exponent: float,
        fill_technique: str):
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
