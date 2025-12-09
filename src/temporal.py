import numpy as np
from collections import deque

class TemporalFilter:
    def __init__(self, buffer_size=8):
        self.buffer_size = buffer_size
        self.left_fit_buffer = deque(maxlen=buffer_size)
        self.right_fit_buffer = deque(maxlen=buffer_size)
        self.left_conf_buffer = deque(maxlen=buffer_size)
        self.right_conf_buffer = deque(maxlen=buffer_size)
        
    def update(self, current_left_fit, current_right_fit, left_conf, right_conf):
        # 🛑 STABILITY FIX: Only update buffer with a valid, high-confidence fit (conf > 0.8)
        if current_left_fit is not None and left_conf > 0.8: 
            self.left_fit_buffer.append(current_left_fit)
        
        if current_right_fit is not None and right_conf > 0.8:
            self.right_fit_buffer.append(current_right_fit)

        # The smoothed fit is the mean of the buffer for maximum stability
        # If the buffer is empty, use the current fit (if it's not None)
        
        if self.left_fit_buffer:
            smoothed_left = np.mean(self.left_fit_buffer, axis=0) 
        else:
            smoothed_left = current_left_fit
            
        if self.right_fit_buffer:
            smoothed_right = np.mean(self.right_fit_buffer, axis=0)
        else:
            smoothed_right = current_right_fit
            
        return smoothed_left, smoothed_right
    
    def get_average_confidence(self):
        """Get average confidence over buffer"""
        left_avg = np.mean(list(self.left_conf_buffer)) if self.left_conf_buffer else 0.0
        right_avg = np.mean(list(self.right_conf_buffer)) if self.right_conf_buffer else 0.0
        return left_avg, right_avg