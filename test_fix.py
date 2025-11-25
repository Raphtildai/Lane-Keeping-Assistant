#!/usr/bin/env python3
"""
Test the fixes for thresholding and perspective transform
"""

import cv2
import numpy as np
import sys
import os

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from preprocess import LanePreprocessor
from warp import PerspectiveTransformer

def test_thresholding():
    """Test the improved thresholding"""
    print("=== TESTING IMPROVED THRESHOLDING ===")
    
    # Create a realistic test image
    width, height = 1280, 720
    img = np.ones((height, width, 3), dtype=np.uint8) * 80  # Dark road
    
    # Add lane markings (bright white on dark road)
    cv2.line(img, (440, 0), (440, height), (255, 255, 255), 12)  # Left lane
    cv2.line(img, (840, 0), (840, height), (255, 255, 255), 12)  # Right lane
    
    # Add some road texture
    noise = np.random.normal(0, 10, (height, width, 3)).astype(np.uint8)
    img = cv2.add(img, noise)
    
    preprocessor = LanePreprocessor()
    result = preprocessor.combine_thresholds(img)
    
    # Save results
    cv2.imwrite("test_input.jpg", img)
    cv2.imwrite("test_threshold_result.jpg", result * 255)
    
    white_pixels = np.sum(result)
    total_pixels = width * height
    percentage = (white_pixels / total_pixels) * 100
    
    print(f"White pixels: {white_pixels} ({percentage:.2f}% of image)")
    
    if percentage < 5:  # Should be a small percentage for lane markings
        print("✅ Thresholding looks good!")
        return True
    else:
        print("❌ Thresholding still detecting too much")
        return False

def test_perspective_transform():
    """Test the improved perspective transform"""
    print("\n=== TESTING PERSPECTIVE TRANSFORM ===")
    
    # Create a test image with lane markings
    width, height = 1280, 720
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Add lane markings (simulating road lanes)
    cv2.line(img, (400, height), (500, height//2), 255, 10)  # Left lane converging
    cv2.line(img, (800, height), (700, height//2), 255, 10)  # Right lane converging
    
    transformer = PerspectiveTransformer((width, height), camera_angle=25.0)
    warped = transformer.warp_to_birdeye(img)
    
    # Save results
    cv2.imwrite("test_perspective_input.jpg", img)
    cv2.imwrite("test_perspective_output.jpg", warped)
    
    input_pixels = np.sum(img > 0)
    output_pixels = np.sum(warped > 0)
    
    print(f"Input white pixels: {input_pixels}")
    print(f"Output white pixels: {output_pixels}")
    
    if output_pixels > 0:
        print("✅ Perspective transform is working!")
        return True
    else:
        print("❌ Perspective transform still losing pixels")
        return False

def test_with_real_video_frame():
    """Test with an actual frame from the video"""
    print("\n=== TESTING WITH REAL VIDEO FRAME ===")
    
    video_path = "sample_test_video.mp4"
    if not os.path.exists(video_path):
        print("Sample video not found")
        return False
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read video frame")
        return False
    
    preprocessor = LanePreprocessor()
    transformer = PerspectiveTransformer(frame.shape[:2][::-1])  # (width, height)
    
    # Test thresholding
    thresholded = preprocessor.combine_thresholds(frame)
    
    # Test perspective transform
    warped = transformer.warp_to_birdeye(thresholded)
    
    # Save results
    cv2.imwrite("real_frame_original.jpg", frame)
    cv2.imwrite("real_frame_thresholded_fixed.jpg", thresholded * 255)
    cv2.imwrite("real_frame_warped_fixed.jpg", warped * 255)
    
    print(f"Original frame size: {frame.shape}")
    print(f"Thresholded white pixels: {np.sum(thresholded)}")
    print(f"Warped white pixels: {np.sum(warped)}")
    
    return np.sum(warped) > 0

if __name__ == "__main__":
    test1 = test_thresholding()
    test2 = test_perspective_transform() 
    test3 = test_with_real_video_frame()
    
    if test1 and test2 and test3:
        print("\n🎉 All tests passed! Now run quick_test.py again.")
    else:
        print("\n❌ Some tests failed. Check the generated images.")