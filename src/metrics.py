import numpy as np
import pandas as pd

# Tracks and reports system performance metrics
# Responsibilities:
#     Data Collection: Records detection results for each frame
#     Lateral Offset Calculation: Measures vehicle position relative to lane center
#     CSV Export: Saves frame-by-frame data for analysis
#     Summary Statistics: Calculates overall performance metrics
#     Stability Measurement: Tracks detection consistency over time
# Key Methods:
#     calculate_lateral_offset(): Computes vehicle offset from lane center
#     record_frame_metrics(): Stores per-frame data
#     save_to_csv(): Exports data for analysis
#     calculate_summary_metrics(): Generates performance reports

class MetricsCalculator:
    def __init__(self):
        self.frame_metrics = []
        
    def calculate_lateral_offset(self, left_fit, right_fit, image_width, image_height):
        """Calculate lateral offset from lane center (in meters)"""
        if left_fit is None or right_fit is None:
            return 0.0
        
        # Calculate lane center at bottom of image
        y = image_height - 1
        left_x = left_fit[0]*y**2 + left_fit[1]*y + left_fit[2]
        right_x = right_fit[0]*y**2 + right_fit[1]*y + right_fit[2]
        lane_center = (left_x + right_x) / 2
        
        # Calculate vehicle center (assumed center of image)
        vehicle_center = image_width / 2
        
        # Convert to meters
        xm_per_pix = 3.7 / 700  # meters per pixel
        offset_pixels = vehicle_center - lane_center
        offset_meters = offset_pixels * xm_per_pix
        
        return offset_meters
    
    def record_frame_metrics(self, frame_id, left_detected, right_detected, 
                           left_conf, right_conf, lat_offset):
        """Record metrics for a single frame"""
        self.frame_metrics.append({
            'frame_id': frame_id,
            'left_detected': int(left_detected),
            'right_detected': int(right_detected),
            'left_conf': left_conf,
            'right_conf': right_conf,
            'lat_offset_m': lat_offset
        })
    
    def save_to_csv(self, filename):
        """Save all frame metrics to CSV"""
        df = pd.DataFrame(self.frame_metrics)
        df.to_csv(filename, index=False)
        return df
    
    def calculate_summary_metrics(self, ground_truth=None):
        """Calculate summary metrics for the entire video"""
        if not self.frame_metrics:
            return {}
        
        df = pd.DataFrame(self.frame_metrics)
        
        summary = {
            'total_frames': len(df),
            'left_detection_rate': df['left_detected'].mean(),
            'right_detection_rate': df['right_detected'].mean(),
            'avg_left_confidence': df['left_conf'].mean(),
            'avg_right_confidence': df['right_conf'].mean(),
            'avg_lateral_offset': df['lat_offset_m'].mean(),
            'lateral_offset_std': df['lat_offset_m'].std()
        }
        
        # Calculate stability metrics
        left_transitions = np.abs(np.diff(df['left_detected'])).sum()
        right_transitions = np.abs(np.diff(df['right_detected'])).sum()
        
        summary['left_flicker_rate'] = left_transitions / len(df)
        summary['right_flicker_rate'] = right_transitions / len(df)
        
        return summary