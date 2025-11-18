#!/usr/bin/env python3
"""
Create synthetic test images for stereo generation
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_test_image():
    """Create a colorful test image with shapes at different depths"""
    width, height = 800, 600
    img = Image.new('RGB', (width, height), color=(220, 230, 240))
    draw = ImageDraw.Draw(img)

    # Background gradient
    for y in range(height):
        color_value = int(180 + (y / height) * 60)
        draw.line([(0, y), (width, y)], fill=(color_value, color_value-20, color_value+20))

    # Draw circles at different depths (for testing)
    # Far circle (will be small in depth map)
    draw.ellipse([150, 150, 300, 300], fill=(100, 150, 200), outline=(50, 100, 150), width=3)

    # Mid circle
    draw.ellipse([350, 200, 550, 400], fill=(200, 100, 100), outline=(150, 50, 50), width=3)

    # Near circle (will be large/bright in depth map)
    draw.ellipse([200, 350, 400, 550], fill=(100, 200, 100), outline=(50, 150, 50), width=3)

    # Add some text
    draw.text((width//2 - 100, 50), "STEREO TEST", fill=(50, 50, 50))

    return img

def create_depth_map():
    """Create corresponding depth map (white=near, black=far)"""
    width, height = 800, 600
    depth = np.zeros((height, width), dtype=np.uint8)

    # Create background gradient (far to mid)
    for y in range(height):
        depth[y, :] = int(80 + (y / height) * 50)

    # Draw depth circles
    img_depth = Image.fromarray(depth)
    draw = ImageDraw.Draw(img_depth)

    # Far circle (dark = far away)
    draw.ellipse([150, 150, 300, 300], fill=100, outline=120, width=5)

    # Mid circle
    draw.ellipse([350, 200, 550, 400], fill=170, outline=190, width=5)

    # Near circle (bright = close)
    draw.ellipse([200, 350, 400, 550], fill=240, outline=255, width=5)

    return img_depth

def main():
    print("Creating synthetic test images...")

    # Create and save test image
    test_img = create_test_image()
    test_img.save("test_image.jpg")
    print("✓ Created test_image.jpg")

    # Create and save depth map
    depth_map = create_depth_map()
    depth_map.save("test_depth.png")
    print("✓ Created test_depth.png")

    print("\nTest images created successfully!")
    print("  - test_image.jpg: 800x600 color image with shapes at different depths")
    print("  - test_depth.png: 800x600 grayscale depth map (white=near, black=far)")

if __name__ == "__main__":
    main()
