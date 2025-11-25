#!/usr/bin/env python3
"""
Minimal test to isolate the perspective transform issue
"""

import cv2
import numpy as np
import os

def create_simple_test_image():
    """Create a simple test image with clear lane markings"""
    width, height = 1280, 720
    img = np.zeros((height, width), dtype=np.uint8)  # Black background
    
    # Add two vertical white lines (simulating lane markings)
    cv2.line(img, (400, 0), (400, height), 255, 20)  # Left lane
    cv2.line(img, (800, 0), (800, height), 255, 20)  # Right lane
    
    cv2.imwrite("simple_test_input.png", img)
    print(f"Created test image with {np.sum(img > 0)} white pixels")
    return img

def test_basic_warp():
    """Test basic OpenCV perspective transform without our class"""
    print("\n=== TESTING BASIC OPENCV WARP ===")
    
    img = create_simple_test_image()
    height, width = img.shape
    
    # Simple perspective points that should definitely work
    src_points = np.float32([
        [300, 500],  # Top-left
        [900, 500],  # Top-right  
        [900, 700],  # Bottom-right
        [300, 700]   # Bottom-left
    ])
    
    dst_points = np.float32([
        [300, 100],  # Top-left
        [900, 100],  # Top-right
        [900, 700],  # Bottom-right  
        [300, 700]   # Bottom-left
    ])
    
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped = cv2.warpPerspective(img, M, (width, height))
    
    print(f"Basic warp: {np.sum(img > 0)} -> {np.sum(warped > 0)} pixels")
    cv2.imwrite("basic_warp_result.png", warped)
    
    return warped

def test_our_warp_class():
    """Test our PerspectiveTransformer class"""
    print("\n=== TESTING OUR PERSPECTIVE TRANSFORMER ===")
    
    # Add src to path
    import sys
    sys.path.append('src')
    
    from warp import PerspectiveTransformer
    
    img = create_simple_test_image()
    transformer = PerspectiveTransformer(img.shape[::-1], camera_angle=25.0)  # (width, height)
    
    warped = transformer.warp_to_birdeye(img)
    print(f"Our transformer: {np.sum(img > 0)} -> {np.sum(warped > 0)} pixels")
    cv2.imwrite("our_warp_result.png", warped)
    
    # Test debug function
    transformer.debug_warp(img, "our_warp_debug.jpg")
    
    return warped

def test_with_real_frame():
    """Test with a real frame from our video"""
    print("\n=== TESTING WITH REAL VIDEO FRAME ===")
    
    video_path = "sample_test_video.mp4"
    if not os.path.exists(video_path):
        print("Video not found")
        return
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read frame")
        return
    
    # Convert to grayscale for simplicity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Threshold to get binary image
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    
    cv2.imwrite("real_frame_binary.png", binary)
    print(f"Real frame binary: {np.sum(binary > 0)} white pixels")
    
    # Test our transformer
    import sys
    sys.path.append('src')
    from warp import PerspectiveTransformer
    
    transformer = PerspectiveTransformer(binary.shape[::-1])
    warped = transformer.warp_to_birdeye(binary)
    
    print(f"Real frame warp: {np.sum(binary > 0)} -> {np.sum(warped > 0)} pixels")
    cv2.imwrite("real_frame_warped.png", warped)
    
    transformer.debug_warp(binary, "real_frame_warp_debug.jpg")

if __name__ == "__main__":
    print("=== MINIMAL PERSPECTIVE TRANSFORM TEST ===")
    
    # Test 1: Basic OpenCV (should work)
    test_basic_warp()
    
    # Test 2: Our class (might fail)
    test_our_warp_class()
    
    # Test 3: Real frame
    test_with_real_frame()
    
    print("\nCheck the generated images to see what's happening!")