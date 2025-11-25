#!/usr/bin/env python3
"""
Test the lane detection WITHOUT perspective transform
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
from lane_fit import LaneDetector
from overlay import OverlayRenderer
from metrics import MetricsCalculator
from utils import apply_roi

class SimpleLaneDetector:
    """Simplified lane detector that skips perspective transform"""
    
    def __init__(self):
        self.preprocessor = LanePreprocessor()
        self.lane_detector = LaneDetector()
        self.overlay_renderer = OverlayRenderer()
        self.metrics_calculator = MetricsCalculator()
    
    def process_video(self, video_path, output_dir):
        """Process video without perspective transform"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_id = 0
        results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Process frame WITHOUT perspective transform
                binary_mask = self.preprocessor.combine_thresholds(frame)
                roi_mask, _ = apply_roi(binary_mask)
                
                # Use ROI mask directly (no perspective transform)
                birdseye = roi_mask
                
                # Lane detection on the ROI
                leftx_base, rightx_base = self.lane_detector.find_lane_base(birdseye)
                
                if leftx_base is None or rightx_base is None:
                    # Use reasonable defaults
                    birdseye_width = birdseye.shape[1]
                    leftx_base, rightx_base = birdseye_width // 4, 3 * birdseye_width // 4
                
                left_inds, right_inds = self.lane_detector.sliding_window_search(
                    birdseye, leftx_base, rightx_base
                )
                
                left_fit, right_fit, leftx, lefty, rightx, righty = self.lane_detector.fit_polynomial(
                    birdseye, left_inds, right_inds
                )
                
                # Calculate confidence (simplified)
                left_conf = 0.8 if left_fit is not None else 0.0
                right_conf = 0.8 if right_fit is not None else 0.0
                
                # Create simple overlay
                overlay = frame.copy()
                if left_fit is not None:
                    ploty = np.linspace(0, height-1, height)
                    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
                    points = np.array([np.transpose(np.vstack([left_fitx, ploty]))], dtype=np.int32)
                    cv2.polylines(overlay, points, False, (0, 255, 0), 5)
                
                if right_fit is not None:
                    ploty = np.linspace(0, height-1, height)
                    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
                    points = np.array([np.transpose(np.vstack([right_fitx, ploty]))], dtype=np.int32)
                    cv2.polylines(overlay, points, False, (255, 0, 0), 5)
                
                # Add HUD
                left_detected = left_conf > 0.5
                right_detected = right_conf > 0.5
                cv2.putText(overlay, f"Left: {'YES' if left_detected else 'NO'}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(overlay, f"Right: {'YES' if right_detected else 'NO'}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                out.write(overlay)
                results.append({
                    'frame_id': frame_id,
                    'left_detected': int(left_detected),
                    'right_detected': int(right_detected),
                    'left_conf': left_conf,
                    'right_conf': right_conf
                })
                
                if frame_id % 30 == 0:
                    print(f"Processed frame {frame_id}")
                
            except Exception as e:
                print(f"Error frame {frame_id}: {e}")
                out.write(frame)  # Write original frame
                results.append({
                    'frame_id': frame_id,
                    'left_detected': 0,
                    'right_detected': 0,
                    'left_conf': 0.0,
                    'right_conf': 0.0
                })
            
            frame_id += 1
        
        cap.release()
        out.release()
        
        # Save results
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(output_dir, 'results.csv'), index=False)
        
        # Calculate summary
        left_detection_rate = df['left_detected'].mean()
        right_detection_rate = df['right_detected'].mean()
        
        print(f"\n=== RESULTS ===")
        print(f"Left detection rate: {left_detection_rate:.2%}")
        print(f"Right detection rate: {right_detection_rate:.2%}")
        print(f"Output saved to: {output_dir}")

if __name__ == "__main__":
    video_path = "sample_test_video.mp4"
    if not os.path.exists(video_path):
        print("Video not found. Please run the test video creation first.")
    else:
        detector = SimpleLaneDetector()
        detector.process_video(video_path, "no_perspective_output")