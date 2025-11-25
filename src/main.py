import cv2
import argparse
import os
import sys

import numpy as np

# Add the current directory to Python path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from preprocess import LanePreprocessor
from warp import PerspectiveTransformer
from lane_fit import LaneDetector
from temporal import TemporalFilter
from overlay import OverlayRenderer
from metrics import MetricsCalculator
from utils import apply_roi

# This is the main driver that coordinates everything
# Responsibilities:
#     Video I/O: Reads input video and writes annotated output
#     Pipeline Coordination: Calls each processing step in sequence
#     Frame Management: Processes each frame through the complete pipeline
#     Component Initialization: Creates all the processing objects
#     Error Handling: Catches and manages errors to prevent complete failure
# Key Components:
#     LKASystem class: Main controller class
#     Video capture and writer setup
#     Frame-by-frame processing loop
#     Metrics collection and output

class LKASystem:
    def __init__(self, video_path, output_dir, debug=False):
        self.video_path = video_path
        self.output_dir = output_dir
        self.debug = debug
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        self.preprocessor = LanePreprocessor()
        
        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.total_frames}")
        
        self.perspective_transformer = PerspectiveTransformer((self.width, self.height))
        self.lane_detector = LaneDetector()
        self.temporal_filter = TemporalFilter()
        self.overlay_renderer = OverlayRenderer()
        self.metrics_calculator = MetricsCalculator()
        
        # Test perspective transform on first frame (with error handling)
        if self.debug:
            try:
                ret, test_frame = self.cap.read()
                if ret:
                    # Reset video position
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    
                    # Test perspective transform
                    binary_mask = self.preprocessor.combine_thresholds(test_frame)
                    
                    # Check if debug_perspective method exists
                    if hasattr(self.perspective_transformer, 'debug_perspective'):
                        self.perspective_transformer.debug_perspective(
                            binary_mask, 
                            os.path.join(output_dir, "perspective_debug.jpg")
                        )
                    else:
                        print("Warning: debug_perspective method not available")
                        # Create a simple debug image instead
                        debug_img = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)
                        cv2.imwrite(os.path.join(output_dir, "simple_debug.jpg"), debug_img)
            except Exception as e:
                print(f"Debug initialization error: {e}")
                # Continue anyway - debug is optional
        
        # Video writer for output
        output_path = os.path.join(output_dir, 'annotated_output.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        if not self.out.isOpened():
            raise ValueError(f"Cannot create output video: {output_path}")
        
    def process_frame(self, frame, frame_id):
        """Process a single frame through the LKA pipeline"""
        try:
            # Preprocessing
            binary_mask = self.preprocessor.combine_thresholds(frame)
            
            # Apply ROI
            roi_mask, roi_offset = apply_roi(binary_mask)
            
            # Perspective transform with fallback
            try:
                birdseye = self.perspective_transformer.warp_to_birdeye(roi_mask)
                
                # If warp returns empty image, use the ROI mask as fallback
                if np.sum(birdseye) == 0:
                    print(f"Frame {frame_id}: Warp failed, using ROI mask as fallback")
                    birdseye = roi_mask
                    
            except Exception as e:
                print(f"Frame {frame_id}: Perspective transform error: {e}")
                birdseye = roi_mask  # Fallback to ROI mask
            
            # Lane detection
            leftx_base, rightx_base = self.lane_detector.find_lane_base(birdseye)
            
            # If no base points found, use reasonable defaults
            if leftx_base is None or rightx_base is None:
                birdseye_width = birdseye.shape[1]
                leftx_base, rightx_base = birdseye_width // 4, 3 * birdseye_width // 4
                print(f"Frame {frame_id}: Using default lane bases: {leftx_base}, {rightx_base}")
            
            left_inds, right_inds = self.lane_detector.sliding_window_search(
                birdseye, leftx_base, rightx_base
            )
            
            left_fit, right_fit, leftx, lefty, rightx, righty = self.lane_detector.fit_polynomial(
                birdseye, left_inds, right_inds
            )
            
            # Calculate confidence
            prev_left_fit = self.temporal_filter.left_fit_buffer[-1] if self.temporal_filter.left_fit_buffer else None
            prev_right_fit = self.temporal_filter.right_fit_buffer[-1] if self.temporal_filter.right_fit_buffer else None
            
            left_conf = self.lane_detector.calculate_confidence(left_fit, leftx, lefty, prev_left_fit)
            right_conf = self.lane_detector.calculate_confidence(right_fit, rightx, righty, prev_right_fit)
            
            # Temporal smoothing
            smoothed_left, smoothed_right = self.temporal_filter.update(left_fit, right_fit, left_conf, right_conf)
            
            # Calculate lateral offset
            # What the Values Mean:
            # lateral_offset = 0.0: Vehicle perfectly centered in lane
            # lateral_offset = +0.2: Vehicle 0.2 meters right of center
            # lateral_offset = -0.3: Vehicle 0.3 meters left of center
            lat_offset = self.metrics_calculator.calculate_lateral_offset(
                smoothed_left, smoothed_right, self.width, self.height
            )
            
            # Detection flags
            left_detected = left_conf > self.overlay_renderer.confidence_threshold
            right_detected = right_conf > self.overlay_renderer.confidence_threshold
            
            # Record metrics
            self.metrics_calculator.record_frame_metrics(
                frame_id, left_detected, right_detected, left_conf, right_conf, lat_offset
            )
            
            # Create overlay
            overlay = self.overlay_renderer.create_overlay(
                frame, smoothed_left, smoothed_right, 
                self.perspective_transformer.Minv, left_conf, right_conf, lat_offset
            )
            
            return overlay
            
        except Exception as e:
            print(f"Error processing frame {frame_id}: {e}")
            # Return original frame if processing fails
            return frame
    
    def run(self):
        """Run the LKA system on the entire video"""
        print(f"Processing video: {self.video_path}")
        
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame, frame_id)
            
            # Write to output video
            self.out.write(processed_frame)
            
            # Display progress
            if frame_id % 30 == 0:
                print(f"Processed frame {frame_id}/{self.total_frames}")
            
            frame_id += 1
        
        # Cleanup
        self.cap.release()
        self.out.release()
        
        # Save metrics
        csv_path = os.path.join(self.output_dir, 'per_frame_metrics.csv')
        self.metrics_calculator.save_to_csv(csv_path)
        
        # Calculate and print summary
        summary = self.metrics_calculator.calculate_summary_metrics()
        print("\n=== Processing Summary ===")
        for key, value in summary.items():
            print(f"{key}: {value:.4f}")
        
        print(f"\nResults saved to: {self.output_dir}")

def main():
    parser = argparse.ArgumentParser(description='Lane Keep Assist System')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, default='outputs', help='Output directory')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video):
        print(f"Error: Video file '{args.video}' not found")
        return
    
    # Create and run LKA system
    lka_system = LKASystem(args.video, args.output, debug=args.debug)
    lka_system.run()

if __name__ == "__main__":
    main()