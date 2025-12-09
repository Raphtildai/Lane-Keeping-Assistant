import numpy as np
import pandas as pd

class MetricsCalculator:
    def __init__(self):
        self.frame_metrics = []
        
    def calculate_lateral_offset(self, left_fit, right_fit, image_width, image_height):
        """Calculate lateral offset from lane center (in meters)"""
        # 🛑 FIX: Ensure fits are not None before attempting to calculate offset
        if left_fit is None or right_fit is None:
            return 0.0
            
        # Check if the fits are valid polynomial arrays (size=3)
        if left_fit.size != 3 or right_fit.size != 3:
            return 0.0
            
        # Calculate lane center at bottom of image
        y = image_height - 1
        left_x = np.polyval(left_fit, y)
        right_x = np.polyval(right_fit, y)
        lane_center = (left_x + right_x) / 2
        
        # Calculate vehicle center (assumed center of image)
        vehicle_center = image_width / 2
        
        # Convert to meters
        xm_per_pix = 3.7 / 700 
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