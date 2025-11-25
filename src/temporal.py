import numpy as np
from collections import deque

# Adds temporal consistency to prevent flickering between frames
# Responsibilities:
#     Frame Buffering: Stores previous lane detections
#     Exponential Smoothing: Smooths lane coefficients over time
#     Stability Enhancement: Reduces jitter and flickering
#     Confidence Tracking: Maintains confidence scores over multiple frames
# Key Methods:
#     smooth_fit(): Applies smoothing to polynomial coefficients
#     update(): Updates the filter with new detections
#     get_average_confidence(): Provides smoothed confidence values
# Why this is important:
#     Prevents lane lines from jumping between frames
#     Provides smoother steering inputs
#     Handles temporary detection failures

class TemporalFilter:
    def __init__(self, buffer_size=5, smooth_factor=0.7):
        self.buffer_size = buffer_size
        self.smooth_factor = smooth_factor
        self.left_fit_buffer = deque(maxlen=buffer_size)
        self.right_fit_buffer = deque(maxlen=buffer_size)
        self.left_conf_buffer = deque(maxlen=buffer_size)
        self.right_conf_buffer = deque(maxlen=buffer_size)
        
    def smooth_fit(self, current_fit, fit_buffer, confidence):
        """Apply exponential smoothing to polynomial coefficients"""
        if current_fit is None:
            if len(fit_buffer) > 0:
                return fit_buffer[-1]  # Return last good fit
            else:
                return None
        
        if len(fit_buffer) == 0:
            fit_buffer.append(current_fit)
            return current_fit
        
        # Weighted average based on confidence
        prev_fit = fit_buffer[-1]
        smoothed_fit = (
            self.smooth_factor * prev_fit + 
            (1 - self.smooth_factor) * current_fit
        )
        
        fit_buffer.append(smoothed_fit)
        return smoothed_fit
    
    def update(self, left_fit, right_fit, left_conf, right_conf):
        """Update temporal filter with new detections"""
        smoothed_left = self.smooth_fit(left_fit, self.left_fit_buffer, left_conf)
        smoothed_right = self.smooth_fit(right_fit, self.right_fit_buffer, right_conf)
        
        self.left_conf_buffer.append(left_conf)
        self.right_conf_buffer.append(right_conf)
        
        return smoothed_left, smoothed_right
    
    def get_average_confidence(self):
        """Get average confidence over buffer"""
        left_avg = np.mean(list(self.left_conf_buffer)) if self.left_conf_buffer else 0.0
        right_avg = np.mean(list(self.right_conf_buffer)) if self.right_conf_buffer else 0.0
        return left_avg, right_avg