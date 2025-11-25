#!/usr/bin/env python3
"""
Script to manually calibrate the perspective transform points
"""

import cv2
import numpy as np
import os
import sys

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from preprocess import LanePreprocessor

def manual_perspective_calibration():
    """Manually find the right perspective points"""
    # Load a frame from the video
    video_path = "sample_test_video.mp4"
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Could not load video frame")
        return
    
    # Create binary mask
    preprocessor = LanePreprocessor()
    binary_mask = preprocessor.combine_thresholds(frame)
    
    height, width = binary_mask.shape
    
    # Test different perspective point configurations
    configurations = [
        {
            'name': 'Wide trapezoid',
            'src': np.float32([
                [width * 0.10, height * 0.95],  # Bottom-left
                [width * 0.90, height * 0.95],  # Bottom-right
                [width * 0.75, height * 0.30],  # Top-right
                [width * 0.25, height * 0.30]   # Top-left
            ])
        },
        {
            'name': 'Narrow trapezoid', 
            'src': np.float32([
                [width * 0.20, height * 0.95],
                [width * 0.80, height * 0.95],
                [width * 0.65, height * 0.30],
                [width * 0.35, height * 0.30]
            ])
        },
        {
            'name': 'Very narrow',
            'src': np.float32([
                [width * 0.30, height * 0.95],
                [width * 0.70, height * 0.95],
                [width * 0.60, height * 0.30],
                [width * 0.40, height * 0.30]
            ])
        }
    ]
    
    # Common destination points
    dst = np.float32([
        [width * 0.20, height],
        [width * 0.80, height],
        [width * 0.80, 0],
        [width * 0.20, 0]
    ])
    
    print("Testing perspective configurations...")
    
    best_config = None
    best_pixels = 0
    
    for config in configurations:
        M = cv2.getPerspectiveTransform(config['src'], dst)
        warped = cv2.warpPerspective(binary_mask, M, (width, height))
        
        pixel_count = np.sum(warped > 0)
        print(f"{config['name']}: {pixel_count} pixels after warp")
        
        # Save visualization
        debug_img = frame.copy()
        
        # Draw the trapezoid
        points = config['src'].astype(np.int32)
        cv2.polylines(debug_img, [points], True, (0, 255, 0), 3)
        
        # Combine with warped view
        warped_bgr = cv2.cvtColor(warped * 255, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([debug_img, warped_bgr])
        
        cv2.imwrite(f"perspective_{config['name'].replace(' ', '_').lower()}.jpg", combined)
        
        if pixel_count > best_pixels:
            best_pixels = pixel_count
            best_config = config
    
    print(f"\nBest configuration: {best_config['name']} with {best_pixels} pixels")
    
    # Test the best configuration with the actual PerspectiveTransformer
    if best_config:
        from warp import PerspectiveTransformer
        
        # Monkey patch the best points into the transformer
        def best_calculate_matrices(self):
            height, width = self.img_size
            src = best_config['src']
            dst = np.float32([
                [width * 0.20, height],
                [width * 0.80, height],
                [width * 0.80, 0],
                [width * 0.20, 0]
            ])
            M = cv2.getPerspectiveTransform(src, dst)
            Minv = cv2.getPerspectiveTransform(dst, src)
            return M, Minv
        
        # Replace the method
        PerspectiveTransformer._calculate_perspective_matrices = best_calculate_matrices
        
        # Test it
        transformer = PerspectiveTransformer((width, height))
        warped = transformer.warp_to_birdeye(binary_mask)
        
        print(f"Final test: {np.sum(warped > 0)} pixels")
        
        # Save final debug
        transformer.debug_perspective(binary_mask, "final_perspective_calibration.jpg")

if __name__ == "__main__":
    manual_perspective_calibration()