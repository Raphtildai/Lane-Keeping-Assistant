#!/usr/bin/env python3
"""
Debug script to identify where the lane detection pipeline is failing
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from preprocess import LanePreprocessor
from import PerspectiveTransformer
from lane_fit import LaneDetector
from utils import apply_roi

def debug_pipeline():
    """Debug each step of the pipeline"""
    print("=== LANE DETECTION PIPELINE DEBUG ===")
    
    # Create a test frame
    width, height = 1280, 720
    frame = np.ones((height, width, 3), dtype=np.uint8) * 80  # Dark gray road
    
    # Add clear lane markings
    center_x = width // 2
    left_lane_x = center_x - 200
    right_lane_x = center_x + 200
    
    # Bright white lane markings
    cv2.line(frame, (left_lane_x, 0), (left_lane_x, height), (255, 255, 255), 12)
    cv2.line(frame, (right_lane_x, 0), (right_lane_x, height), (255, 255, 255), 12)
    
    print("1. Original frame created with lane markings")
    print(f"   Left lane at x={left_lane_x}, Right lane at x={right_lane_x}")
    
    # Step 1: Preprocessing
    print("\n2. Testing preprocessing...")
    preprocessor = LanePreprocessor()
    
    # Test individual threshold methods
    s_binary = preprocessor.hls_select(frame)
    sobel_binary = preprocessor.sobel_edges(frame)
    white_binary = preprocessor.white_color_threshold(frame)
    color_binary = preprocessor.combined_color_threshold(frame)
    combined = preprocessor.combine_thresholds(frame)
    
    print(f"   S channel: {np.sum(s_binary)} white pixels")
    print(f"   Sobel: {np.sum(sobel_binary)} white pixels") 
    print(f"   White threshold: {np.sum(white_binary)} white pixels")
    print(f"   Color threshold: {np.sum(color_binary)} white pixels")
    print(f"   Combined: {np.sum(combined)} white pixels")
    
    if np.sum(combined) == 0:
        print("   ❌ ERROR: Combined threshold has no white pixels!")
        print("   The lane markings are not being detected at all.")
        return False
    
    # Step 2: ROI
    print("\n3. Testing ROI application...")
    roi_mask, roi_offset = apply_roi(combined)
    print(f"   ROI applied: {np.sum(roi_mask)} white pixels in ROI")
    
    if np.sum(roi_mask) == 0:
        print("   ❌ ERROR: No white pixels in ROI!")
        return False
    
    # Step 3: Perspective transform
    print("\n4. Testing perspective transform...")
    perspective_transformer = PerspectiveTransformer((width, height))
    birdseye = perspective_transformer.warp_to_birdeye(roi_mask)
    print(f"   Birdseye view: {np.sum(birdseye)} white pixels")
    
    if np.sum(birdseye) == 0:
        print("   ❌ ERROR: No white pixels in birdseye view!")
        return False
    
    # Step 4: Lane detection
    print("\n5. Testing lane detection...")
    lane_detector = LaneDetector()
    
    # Find lane bases
    leftx_base, rightx_base = lane_detector.find_lane_base(birdseye)
    print(f"   Lane bases - Left: {leftx_base}, Right: {rightx_base}")
    
    if leftx_base is None or rightx_base is None:
        print("   ❌ ERROR: Could not find lane bases!")
        print("   Trying manual lane base positions...")
        leftx_base, rightx_base = width // 4, 3 * width // 4
        print(f"   Using manual bases - Left: {leftx_base}, Right: {rightx_base}")
    
    # Sliding window search
    left_inds, right_inds = lane_detector.sliding_window_search(
        birdseye, leftx_base, rightx_base
    )
    
    print(f"   Left lane pixels found: {len(left_inds)}")
    print(f"   Right lane pixels found: {len(right_inds)}")
    
    if len(left_inds) == 0 and len(right_inds) == 0:
        print("   ❌ ERROR: No lane pixels found in sliding window search!")
        return False
    
    # Polynomial fitting
    left_fit, right_fit, leftx, lefty, rightx, righty = lane_detector.fit_polynomial(
        birdseye, left_inds, right_inds
    )
    
    print(f"   Left fit: {left_fit}")
    print(f"   Right fit: {right_fit}")
    print(f"   Left lane points: {len(leftx)}")
    print(f"   Right lane points: {len(rightx)}")
    
    if left_fit is None and right_fit is None:
        print("   ❌ ERROR: Could not fit polynomials to lane pixels!")
        return False
    
    # Step 5: Confidence calculation
    print("\n6. Testing confidence calculation...")
    left_conf = lane_detector.calculate_confidence(left_fit, leftx, lefty, None)
    right_conf = lane_detector.calculate_confidence(right_fit, rightx, righty, None)
    
    print(f"   Left confidence: {left_conf:.3f}")
    print(f"   Right confidence: {right_conf:.3f}")
    
    # Step 6: Create visualization
    print("\n7. Creating debug visualization...")
    
    # Create a debug image showing all steps
    debug_height = height * 2
    debug_width = width * 2
    debug_img = np.zeros((debug_height, debug_width, 3), dtype=np.uint8)
    
    # Top row: Original and thresholded images
    debug_img[0:height, 0:width] = frame
    debug_img[0:height, width:width*2] = cv2.cvtColor(combined * 255, cv2.COLOR_GRAY2BGR)
    
    # Bottom row: ROI and birdseye
    roi_visual = cv2.cvtColor(roi_mask * 255, cv2.COLOR_GRAY2BGR)
    debug_img[height:height*2, 0:width] = roi_visual
    
    birdseye_visual = cv2.cvtColor(birdseye * 255, cv2.COLOR_GRAY2BGR)
    debug_img[height:height*2, width:width*2] = birdseye_visual
    
    # Add lane detection results to birdseye view
    if left_fit is not None:
        ploty = np.linspace(0, birdseye.shape[0]-1, birdseye.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        pts = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        cv2.polylines(birdseye_visual, np.int32(pts), False, (0, 255, 0), 3)
    
    if right_fit is not None:
        ploty = np.linspace(0, birdseye.shape[0]-1, birdseye.shape[0])
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        pts = np.array([np.transpose(np.vstack([right_fitx, ploty]))])
        cv2.polylines(birdseye_visual, np.int32(pts), False, (255, 0, 0), 3)
    
    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(debug_img, "Original", (10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(debug_img, "Combined Threshold", (width + 10, 30), font, 1, (255, 255, 255), 2)
    cv2.putText(debug_img, "ROI Mask", (10, height + 30), font, 1, (255, 255, 255), 2)
    cv2.putText(debug_img, "Birdseye + Lanes", (width + 10, height + 30), font, 1, (255, 255, 255), 2)
    
    # Save debug image
    debug_path = "pipeline_debug.jpg"
    cv2.imwrite(debug_path, debug_img)
    print(f"   Debug visualization saved: {debug_path}")
    
    print("\n✅ Pipeline debug completed!")
    print("Check the generated image to see where the issue occurs.")
    return True

def test_with_real_frame():
    """Test with an actual frame from the video"""
    print("\n=== TESTING WITH ACTUAL VIDEO FRAME ===")
    
    video_path = "sample_test_video.mp4"
    if not os.path.exists(video_path):
        print("Sample video not found. Run quick_test.py first.")
        return
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not read frame from video.")
        return
    
    print("Frame loaded from video. Testing pipeline...")
    
    # Test the preprocessing on the real frame
    preprocessor = LanePreprocessor()
    combined = preprocessor.combine_thresholds(frame)
    
    print(f"White pixels in real frame: {np.sum(combined)}")
    
    # Save the original frame and thresholded result
    cv2.imwrite("real_frame_original.jpg", frame)
    cv2.imwrite("real_frame_thresholded.jpg", combined * 255)
    
    print("Real frame analysis saved to:")
    print("  - real_frame_original.jpg")
    print("  - real_frame_thresholded.jpg")

if __name__ == "__main__":
    debug_pipeline()
    test_with_real_frame()