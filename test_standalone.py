#!/usr/bin/env python3
"""
Standalone test script for ComfyStereo
Tests stereo image generation without ComfyUI dependencies
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from PIL import Image
import stereoimage_generation as sig

def load_and_prepare_images():
    """Load local test images"""
    # Check if images exist
    if not os.path.exists("test_image.jpg") or not os.path.exists("test_depth.png"):
        print("Creating test images...")
        os.system("python create_test_images.py")

    print("Loading test images...")

    # Load images
    original_img = Image.open("test_image.jpg")
    depth_img = Image.open("test_depth.png")

    # Convert to numpy arrays
    original_np = np.array(original_img)
    depth_np = np.array(depth_img)

    # Convert depth to grayscale if needed
    if len(depth_np.shape) == 3:
        depth_np = np.dot(depth_np[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.uint8)

    # Resize depth map to match image if needed
    if original_np.shape[:2] != depth_np.shape:
        depth_img = Image.fromarray(depth_np).resize((original_np.shape[1], original_np.shape[0]))
        depth_np = np.array(depth_img)

    print(f"✓ Image loaded: {original_np.shape}")
    print(f"✓ Depth map loaded: {depth_np.shape}")

    return original_np, depth_np

def test_basic_stereo():
    """Test basic stereo generation"""
    print("\n" + "="*60)
    print("TEST 1: Basic Stereo Generation (Default Settings)")
    print("="*60)

    original, depth = load_and_prepare_images()

    # Generate stereo with default settings
    results, modified_depth = sig.create_stereoimages(
        original_image=original,
        depthmap=depth,
        divergence=3.5,
        separation=0,
        modes=['left-right'],
        stereo_balance=0.0,
        stereo_offset_exponent=1.0,
        fill_technique='polylines_soft',
        depth_blur_strength=0,
        depth_blur_edge_threshold=6.0,
        direction_aware_depth_blur=False,
        return_modified_depth=True
    )

    # Save result
    results[0].save("output_basic.jpg")
    print("✓ Generated stereo image: output_basic.jpg")
    print(f"  Output size: {results[0].size}")

    return True

def test_divergence_levels():
    """Test different divergence levels"""
    print("\n" + "="*60)
    print("TEST 2: Different Divergence Levels")
    print("="*60)

    original, depth = load_and_prepare_images()

    divergence_tests = [
        (1.0, "subtle", "Subtle 3D effect"),
        (3.5, "normal", "Normal 3D effect"),
        (8.0, "strong", "Strong 3D effect"),
    ]

    for div_val, name, description in divergence_tests:
        print(f"\n  Testing divergence={div_val} ({description})...")

        results, _ = sig.create_stereoimages(
            original_image=original,
            depthmap=depth,
            divergence=div_val,
            separation=0,
            modes=['left-right'],
            stereo_balance=0.0,
            stereo_offset_exponent=1.0,
            fill_technique='polylines_soft',
            depth_blur_strength=0,
            depth_blur_edge_threshold=6.0,
            direction_aware_depth_blur=False,
            return_modified_depth=True
        )

        filename = f"output_divergence_{name}.jpg"
        results[0].save(filename)
        print(f"  ✓ Saved: {filename}")

    return True

def test_depth_blur():
    """Test depth map blurring"""
    print("\n" + "="*60)
    print("TEST 3: Depth Map Blurring")
    print("="*60)

    original, depth = load_and_prepare_images()

    print("\n  Testing WITH depth blur...")
    results, left_depth, right_depth = sig.create_stereoimages(
        original_image=original,
        depthmap=depth,
        divergence=5.0,  # Higher divergence to see blur effect
        separation=0,
        modes=['left-right'],
        stereo_balance=0.0,
        stereo_offset_exponent=1.0,
        fill_technique='polylines_soft',
        depth_blur_strength=5.0,
        depth_blur_edge_threshold=6.0,
        direction_aware_depth_blur=True,
        return_modified_depth=True,
    )

    results[0].save("output_with_blur.jpg")
    left_depth.save("output_depth_left.png")
    right_depth.save("output_depth_right.png")

    print("  ✓ Saved: output_with_blur.jpg")
    print("  ✓ Saved: output_depth_left.png")
    print("  ✓ Saved: output_depth_right.png")

    return True

def test_stereo_balance():
    """Test stereo balance parameter"""
    print("\n" + "="*60)
    print("TEST 4: Stereo Balance Control")
    print("="*60)

    original, depth = load_and_prepare_images()

    balance_tests = [
        (-0.8, "right_heavy", "Right eye heavy"),
        (0.0, "balanced", "Balanced"),
        (0.8, "left_heavy", "Left eye heavy"),
    ]

    for balance_val, name, description in balance_tests:
        print(f"\n  Testing balance={balance_val} ({description})...")

        results, _ = sig.create_stereoimages(
            original_image=original,
            depthmap=depth,
            divergence=3.5,
            separation=0,
            modes=['left-right'],
            stereo_balance=balance_val,
            stereo_offset_exponent=1.0,
            fill_technique='polylines_soft',
            depth_blur_strength=0,
            depth_blur_edge_threshold=6.0,
            direction_aware_depth_blur=False,
            return_modified_depth=True,
        )

        filename = f"output_balance_{name}.jpg"
        results[0].save(filename)
        print(f"  ✓ Saved: {filename}")

    return True

def test_different_modes():
    """Test different output modes"""
    print("\n" + "="*60)
    print("TEST 5: Different Output Modes")
    print("="*60)

    original, depth = load_and_prepare_images()

    modes_to_test = [
        ('left-right', 'lr'),
        ('top-bottom', 'tb'),
        ('red-cyan-anaglyph', 'anaglyph'),
    ]

    for mode, suffix in modes_to_test:
        print(f"\n  Testing mode: {mode}...")

        results, _ = sig.create_stereoimages(
            original_image=original,
            depthmap=depth,
            divergence=3.5,
            separation=0,
            modes=[mode],
            stereo_balance=0.0,
            stereo_offset_exponent=1.0,
            fill_technique='polylines_soft',
            depth_blur_strength=0,
            depth_blur_edge_threshold=6.0,
            direction_aware_depth_blur=False,
            return_modified_depth=True,
        )

        filename = f"output_mode_{suffix}.jpg"
        results[0].save(filename)
        print(f"  ✓ Saved: {filename}")

    return True

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("ComfyStereo Standalone Test Suite")
    print("="*60)

    tests = [
        ("Basic Stereo Generation", test_basic_stereo),
        ("Different Divergence Levels", test_divergence_levels),
        ("Depth Map Blurring", test_depth_blur),
        ("Stereo Balance Control", test_stereo_balance),
        ("Different Output Modes", test_different_modes),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {test_name}")

    total = len(results)
    passed = sum(1 for _, s in results if s)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed successfully!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1

if __name__ == "__main__":
    exit(main())
