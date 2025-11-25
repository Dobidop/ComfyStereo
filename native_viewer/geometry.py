"""
Geometry creation functions for different projection types
"""

import math
import numpy as np


def create_sphere_mesh(radius=100.0, segments=60, rings=40):
    """Create sphere geometry for 360° viewing"""
    vertices = []
    indices = []

    # Generate vertices
    for ring in range(rings + 1):
        theta = ring * math.pi / rings
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for seg in range(segments + 1):
            phi = seg * 2 * math.pi / segments
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)

            # Position
            x = radius * sin_theta * cos_phi
            y = radius * cos_theta
            z = radius * sin_theta * sin_phi

            # UV coordinates (flipped for inside viewing)
            u = 1.0 - (seg / segments)
            v = ring / rings

            vertices.extend([x, y, z, u, v])

    # Generate indices
    for ring in range(rings):
        for seg in range(segments):
            first = ring * (segments + 1) + seg
            second = first + segments + 1

            # Two triangles per quad
            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_flat_screen(width=3.0, height=2.25, distance=3.0, horizontal_offset=0.0, vertical_offset=0.0):
    """Create flat screen geometry (like a cinema screen in VR)"""
    vertices = []
    indices = []

    # Calculate aspect ratio preserving dimensions
    half_width = width / 2.0
    half_height = height / 2.0

    # Create a simple quad facing the viewer
    # Position the screen at the given distance
    z = -distance

    # In VR, y=0 is typically floor level, eye level is around 1.5-1.7m
    # Position screen center at typical seated eye height (~1.2m)
    eye_height = 1.2
    y_offset = eye_height + vertical_offset

    # Horizontal offset for user adjustment
    x_offset = horizontal_offset

    # Four corners of the screen (centered at eye height)
    positions = [
        [-half_width + x_offset, -half_height + y_offset, z],  # Bottom left
        [half_width + x_offset, -half_height + y_offset, z],   # Bottom right
        [half_width + x_offset, half_height + y_offset, z],    # Top right
        [-half_width + x_offset, half_height + y_offset, z],   # Top left
    ]

    # UV coordinates
    uvs = [
        [0.0, 1.0],  # Bottom left
        [1.0, 1.0],  # Bottom right
        [1.0, 0.0],  # Top right
        [0.0, 0.0],  # Top left
    ]

    # Build vertices
    for i in range(4):
        vertices.extend(positions[i])
        vertices.extend(uvs[i])

    # Two triangles to form the quad
    indices = [0, 1, 2, 0, 2, 3]

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_curved_screen(width=3.0, height=2.25, distance=3.0, curve_amount=0.3,
                        horizontal_offset=0.0, vertical_offset=0.0):
    """Create curved screen geometry (gently curved like IMAX)"""
    vertices = []
    indices = []

    segments_h = 20  # Horizontal segments for curvature
    segments_v = 10  # Vertical segments

    half_height = height / 2.0

    # In VR, y=0 is typically floor level, eye level is around 1.5-1.7m
    # Position screen center at typical seated eye height (~1.2m)
    eye_height = 1.2
    y_offset = eye_height + vertical_offset

    # Horizontal offset for user adjustment
    x_offset = horizontal_offset

    for v in range(segments_v + 1):
        # Center screen at eye height
        y = -half_height + (v / segments_v) * height + y_offset
        v_uv = 1.0 - (v / segments_v)

        for h in range(segments_h + 1):
            # Create curve using an arc
            angle = (h / segments_h - 0.5) * math.pi * curve_amount
            x = distance * math.sin(angle)
            z = -distance * math.cos(angle)

            # Scale x based on desired width
            x = x * (width / (2.0 * distance * math.sin(math.pi * curve_amount / 2.0)))

            # Apply horizontal offset
            x = x + x_offset

            u = h / segments_h

            vertices.extend([x, y, z, u, v_uv])

    # Generate indices
    for v in range(segments_v):
        for h in range(segments_h):
            first = v * (segments_h + 1) + h
            second = first + segments_h + 1

            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)


def create_dome_180(radius=10.0, segments=60):
    """Create 180° dome/hemisphere geometry"""
    vertices = []
    indices = []

    rings = segments // 2

    # Generate only the front hemisphere
    for ring in range(rings + 1):
        theta = ring * (math.pi / 2) / rings  # 0 to π/2 (front hemisphere)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)

        for seg in range(segments + 1):
            phi = seg * math.pi / segments  # 0 to π (180 degrees horizontally)
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)

            # Position
            x = radius * sin_theta * sin_phi
            y = radius * cos_theta
            z = -radius * sin_theta * cos_phi

            # UV coordinates
            u = seg / segments
            v = ring / rings

            vertices.extend([x, y, z, u, v])

    # Generate indices
    for ring in range(rings):
        for seg in range(segments):
            first = ring * (segments + 1) + seg
            second = first + segments + 1

            indices.extend([first, second, first + 1])
            indices.extend([second, second + 1, first + 1])

    return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.uint32)
