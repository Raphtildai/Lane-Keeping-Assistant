import cv2
import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
import json

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from preprocess import LanePreprocessor
from warp import PerspectiveTransformer
from lane_fit import LaneDetector
from temporal import TemporalFilter
from overlay import OverlayRenderer
from metrics import MetricsCalculator
from utils import apply_roi

class TusimpleLKASystem:
    def __init__(self, base_dir="tusimple_data"):
        self.base_dir = Path(base_dir)
        self.test_videos_dir = self.base_dir / "videos"
        
        # Initialize LKA components
        self.preprocessor = LanePreprocessor()
        self.lane_detector = LaneDetector()
        self.overlay_renderer = OverlayRenderer()
        self.metrics_calculator = MetricsCalculator()
        
    def process_video(self, video_path):
        """Process a single video through LKA pipeline"""
        print(f"Processing: {video_path.name}")
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error opening video: {video_path}")
            return
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Initialize components that need image dimensions
        self.perspective_transformer = PerspectiveTransformer((width, height))
        self.temporal_filter = TemporalFilter()
        
        # Setup output
        output_dir = Path("outputs") / video_path.stem
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_video_path = output_dir / "lka_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))
        
        frame_id = 0
        results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Process frame
                processed_frame, metrics = self.process_single_frame(frame, frame_id)
                out.write(processed_frame)
                results.append(metrics)
                
                if frame_id % 30 == 0:
                    print(f"  Frame {frame_id}")
                    
            except Exception as e:
                print(f"Error processing frame {frame_id}: {e}")
                # Write original frame as fallback
                out.write(frame)
                results.append({
                    'frame_id': frame_id,
                    'left_detected': 0,
                    'right_detected': 0,
                    'left_conf': 0.0,
                    'right_conf': 0.0,
                    'lat_offset_m': 0.0
                })
            
            frame_id += 1
        
        # Cleanup
        cap.release()
        out.release()
        
        # Save results
        self.save_results(results, output_dir)
        print(f"  ✓ Completed: {output_dir}")
    
    def process_single_frame(self, frame, frame_id):
        """Process a single frame"""
        from utils import apply_roi
        
        # Preprocessing
        binary_mask = self.preprocessor.combine_thresholds(frame)
        
        # Apply ROI
        roi_mask, _ = apply_roi(binary_mask)
        
        # Perspective transform
        birdseye = self.perspective_transformer.warp_to_birdeye(roi_mask)
        
        # Lane detection
        try:
            leftx_base, rightx_base = self.lane_detector.find_lane_base(birdseye)
            left_inds, right_inds = self.lane_detector.sliding_window_search(
                birdseye, leftx_base, rightx_base
            )
            
            left_fit, right_fit, leftx, lefty, rightx, righty = self.lane_detector.fit_polynomial(
                birdseye, left_inds, right_inds
            )
        except Exception as e:
            left_fit = right_fit = None
            leftx = lefty = rightx = righty = []
        
        # Calculate confidence
        prev_left_fit = self.temporal_filter.left_fit_buffer[-1] if self.temporal_filter.left_fit_buffer else None
        prev_right_fit = self.temporal_filter.right_fit_buffer[-1] if self.temporal_filter.right_fit_buffer else None
        
        left_conf = self.lane_detector.calculate_confidence(left_fit, leftx, lefty, prev_left_fit)
        right_conf = self.lane_detector.calculate_confidence(right_fit, rightx, righty, prev_right_fit)
        
        # Temporal smoothing
        smoothed_left, smoothed_right = self.temporal_filter.update(left_fit, right_fit, left_conf, right_conf)
        
        # Calculate lateral offset
        lat_offset = self.metrics_calculator.calculate_lateral_offset(
            smoothed_left, smoothed_right, frame.shape[1], frame.shape[0]
        )
        
        # Detection flags
        left_detected = left_conf > self.overlay_renderer.confidence_threshold
        right_detected = right_conf > self.overlay_renderer.confidence_threshold
        
        # Create overlay
        overlay = self.overlay_renderer.create_overlay(
            frame, smoothed_left, smoothed_right, 
            self.perspective_transformer.Minv, left_conf, right_conf, lat_offset
        )
        
        # Prepare metrics
        metrics = {
            'frame_id': frame_id,
            'left_detected': int(left_detected),
            'right_detected': int(right_detected),
            'left_conf': left_conf,
            'right_conf': right_conf,
            'lat_offset_m': lat_offset
        }
        
        return overlay, metrics
    
    def save_results(self, results, output_dir):
        """Save processing results"""
        df = pd.DataFrame(results)
        csv_path = output_dir / "detection_metrics.csv"
        df.to_csv(csv_path, index=False)
        
        # Calculate summary
        summary = self.calculate_summary_metrics(df)
        summary_path = output_dir / "summary_metrics.json"
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"  Summary - Left: {summary['left_detection_rate']:.2%}, "
              f"Right: {summary['right_detection_rate']:.2%}, "
              f"Offset: {summary['avg_lateral_offset']:.3f}m")
    
    def calculate_summary_metrics(self, df):
        """Calculate performance metrics"""
        if len(df) == 0:
            return {}
        
        summary = {
            'total_frames': len(df),
            'left_detection_rate': df['left_detected'].mean(),
            'right_detection_rate': df['right_detected'].mean(),
            'avg_left_confidence': df['left_conf'].mean(),
            'avg_right_confidence': df['right_conf'].mean(),
            'avg_lateral_offset': df['lat_offset_m'].mean(),
            'lateral_offset_std': df['lat_offset_m'].std()
        }
        
        return summary
    
    def run_on_all_test_videos(self):
        """Run LKA on all available test videos"""
        test_videos = list(self.test_videos_dir.glob("*.mp4"))
        
        if not test_videos:
            print("No test videos found. Creating sample video...")
            self._create_fallback_video()
            test_videos = list(self.test_videos_dir.glob("*.mp4"))
        
        print(f"Found {len(test_videos)} test video(s)")
        
        for video_path in test_videos:
            self.process_video(video_path)
    
    def _create_fallback_video(self):
        """Create a fallback video if none exist"""
        from tusimple_downloader import TuSimpleDatasetLoader
        loader = TuSimpleDatasetLoader()
        loader._create_sample_video(self.test_videos_dir / "fallback_video.mp4")

def main():
    lka_system = TusimpleLKASystem()
    lka_system.run_on_all_test_videos()

if __name__ == "__main__":
    main()