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

class LKASystem:
    def __init__(self, video_path, output_dir, debug=True):
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

        # ⚠️ ADD GENERIC CAMERA CALIBRATION DATA (CRITICAL FIX) ⚠️
        # These parameters are typical for a wide-angle dashcam (1280x720) 
        # and help correct the worst of the lens distortion.
        # Focal length (fx, fy) around 1100-1200 is common for good resolution.
        self.mtx = np.array([
            [1100.0, 0.0, self.width / 2],
            [0.0, 1100.0, self.height / 2],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        # Distortion coefficients (k1, k2, p1, p2, k3)
        # Using small values for k1/k2 to handle common barrel distortion.
        self.dist = np.array([-0.24, 0.01, 0.001, -0.001, 0.0], dtype=np.float32)

        print(f"Video properties: {self.width}x{self.height}, FPS: {self.fps}, Frames: {self.total_frames}")
        
        self.perspective_transformer = PerspectiveTransformer((self.width, self.height))
        self.lane_detector = LaneDetector()
        self.temporal_filter = TemporalFilter()
        self.overlay_renderer = OverlayRenderer()
        self.metrics_calculator = MetricsCalculator()
        
        # Test perspective transform on first frame (with error handling)
        if self.debug:
            # VISUALIZE THE PERSPECTIVE POINTS ON THE UNDISTORTED FRAME
            # This will show the green trapezoid (Source) and the blue rectangle (Destination)
            undistorted_frame = self.perspective_transformer.visualize_perspective_areas(undistorted_frame)
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
            # 1. UNDISTORTION 
            undistorted_frame = cv2.undistort(frame, self.mtx, self.dist, None, self.mtx)
            
            # Visualization of perspective points (Recommended for debugging transformation failure)
            if self.debug:
                undistorted_frame = self.perspective_transformer.visualize_perspective_areas(undistorted_frame)
                
            # Preprocessing
            binary_mask = self.preprocessor.combine_thresholds(undistorted_frame)
            
            # Preprocessing
            # Use the undistorted frame for all subsequent processing
            binary_mask = self.preprocessor.combine_thresholds(undistorted_frame)
            
            # Apply ROI (Original frame is now corrected)
            # roi_mask, roi_offset = apply_roi(binary_mask)
            
            # Perspective transform with fallback
            try:
                birdseye = self.perspective_transformer.warp_to_birdeye(binary_mask)
                
                # If warp returns empty image, use the ROI mask as fallback
                if np.sum(birdseye) == 0:
                    print(f"Frame {frame_id}: Warp failed, using ROI mask as fallback")
                    birdseye = binary_mask
                    
            except Exception as e:
                print(f"Frame {frame_id}: Perspective transform error: {e}")
                birdseye = binary_mask  # Fallback to ROI mask
                # Fallback for display
                birdseye_display = np.zeros_like(undistorted_frame)
            
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
            
            # left_conf = self.lane_detector.calculate_confidence(left_fit, leftx, lefty, prev_left_fit)
            # right_conf = self.lane_detector.calculate_confidence(right_fit, rightx, righty, prev_right_fit)
            # # Temporal smoothing
            # smoothed_left, smoothed_right = self.temporal_filter.update(left_fit, right_fit, left_conf, right_conf)
            
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

            # 3. Bird's-Eye View with Detection Overlay
            # You will need a method in LaneDetector (or a utility function) 
            # to draw the fitted polynomial lines onto the birdseye image.
            birdseye_fit_overlay = self.lane_detector.draw_detection(
                birdseye, leftx, lefty, rightx, righty, 
                smoothed_left, smoothed_right
            )
            
            # Create overlay
            overlay = self.overlay_renderer.create_overlay(
                frame, smoothed_left, smoothed_right, 
                self.perspective_transformer.Minv, left_conf, right_conf, lat_offset
            )

            # --- VISUALIZATION STITCHING (Optimized) ---
            if self.debug:
                # --- CONFIGURATION: ADJUST THIS VALUE ---
                # Try 0.3 for a small screen, 0.4 for medium, or 0.5 for large.
                # E.g., for a 1920x1080 video, 0.4 scale makes the window approx 1536x864.
                FINAL_SCALE_FACTOR = 0.4
                
                # 1. Create Birdseye Overlay
                birdseye_fit_overlay = self.lane_detector.draw_detection(
                    birdseye, leftx, lefty, rightx, righty, 
                    smoothed_left, smoothed_right
                )
                
                # 2. Prepare other views for stitching
                h, w, _ = undistorted_frame.shape
                binary_mask_bgr = cv2.cvtColor(binary_mask * 255, cv2.COLOR_GRAY2BGR)

                # Ensure all four images are the same size (Original size for simplicity)
                # You might need to resize birdseye_fit_overlay if it was created with different dims
                # We assume birdseye and undistorted_frame are the same size (w, h)
                
                # 3. Create the 2x2 grid at the FULL SIZE
                top_row = np.hstack((undistorted_frame, binary_mask_bgr))
                bottom_row = np.hstack((birdseye_fit_overlay, overlay))

                debug_visualization = np.vstack((top_row, bottom_row))
                
                # 4. APPLY FINAL RESIZE
                new_width = int(debug_visualization.shape[1] * FINAL_SCALE_FACTOR)
                new_height = int(debug_visualization.shape[0] * FINAL_SCALE_FACTOR)
                
                # Resize the entire stitched image down to fit the screen
                debug_visualization_scaled = cv2.resize(
                    debug_visualization, (new_width, new_height), interpolation=cv2.INTER_LINEAR
                )

                # 5. Show the scaled visualization
                cv2.imshow("LKA Debug Pipeline", debug_visualization_scaled)
                cv2.waitKey(1)
                
            return overlay
            
        except Exception as e:
            # Log the error, but still return a frame to avoid crashing the video writer
            print(f"Error processing frame {frame_id}: {e}")
            # Ensure windows are closed on critical error if needed
            # cv2.destroyAllWindows() 
            return frame # Return original frame if processing fails
    
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
            
            # If not in debug mode, you can still show the final output
            if not self.debug:
                 cv2.imshow("LKA Final Output", processed_frame)
                 if cv2.waitKey(1) & 0xFF == ord('q'):
                     break
            
            # Write to output video
            self.out.write(processed_frame)
            
            # Display progress
            if frame_id % 30 == 0:
                print(f"Processed frame {frame_id}/{self.total_frames}")
            
            frame_id += 1
        
        # Cleanup
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        
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