import numpy as np
import cv2

# Finds lane pixels and fits mathematical curves to represent lanes
# Responsibilities:
#     Histogram Analysis: Finds lane starting positions at the bottom of the image
#     Sliding Window Search: Moves upward to collect lane pixels
#     Polynomial Fitting: Fits 2nd-order polynomials to lane pixels
#     Confidence Calculation: Determines how reliable the detection is
#     Pixel Management: Tracks left and right lane pixels separately
# Key Methods:
#     find_lane_base(): Uses histogram to find initial lane positions
#     sliding_window_search(): Follows lane pixels upward
#     fit_polynomial(): Fits curves to lane pixels
#     calculate_confidence(): Scores detection reliability

class LaneDetector:
    def __init__(self, n_windows=9, margin=100, minpix=50):
        self.n_windows = n_windows
        self.margin = margin
        self.minpix = minpix
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        
    def find_lane_base(self, binary_warped):
        """Find starting position for left and right lanes using histogram"""
        if binary_warped is None or binary_warped.size == 0:
            return None, None
            
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_warped[binary_warped.shape[0]//2:, :], axis=0)
        
        if np.max(histogram) == 0:
            print("No lane pixels found in histogram")
            return None, None
            
        midpoint = histogram.shape[0] // 2
        
        # Find the peak of the left and right halves of the histogram
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # print(f"Histogram peaks - Left: {leftx_base}, Right: {rightx_base}")
        return leftx_base, rightx_base
    
    def sliding_window_search(self, binary_warped, leftx_base, rightx_base):
        """Find lane pixels using sliding window search"""
        if binary_warped is None or leftx_base is None or rightx_base is None:
            return [], []
            
        height = binary_warped.shape[0]
        window_height = height // self.n_windows
        
        # Identify all non-zero pixels
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Current positions to be updated
        leftx_current = leftx_base
        rightx_current = rightx_base
        
        # Lists to receive lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        for window in range(self.n_windows):
            # Window boundaries
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            
            win_xleft_low = max(0, leftx_current - self.margin)
            win_xleft_high = min(binary_warped.shape[1], leftx_current + self.margin)
            win_xright_low = max(0, rightx_current - self.margin)
            win_xright_high = min(binary_warped.shape[1], rightx_current + self.margin)
            
            # Identify nonzero pixels in the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
            # Recenter next window if enough pixels found
            if len(good_left_inds) > self.minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        
        # Concatenate arrays of indices
        if left_lane_inds:
            left_lane_inds = np.concatenate(left_lane_inds)
        else:
            left_lane_inds = np.array([], dtype=np.int64)
            
        if right_lane_inds:
            right_lane_inds = np.concatenate(right_lane_inds)
        else:
            right_lane_inds = np.array([], dtype=np.int64)
        
        print(f"Found {len(left_lane_inds)} left lane pixels, {len(right_lane_inds)} right lane pixels")
        return left_lane_inds, right_lane_inds
    
    def fit_polynomial(self, binary_warped, left_lane_inds, right_lane_inds):
        """Fit polynomials to lane pixels"""
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit second order polynomial
        left_fit = None
        right_fit = None
        
        if len(leftx) > 0 and len(lefty) > 0:
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                # print(f"Left fit coefficients: {left_fit}")
            except:
                print("Failed to fit left lane polynomial")
                left_fit = None
                
        if len(rightx) > 0 and len(righty) > 0:
            try:
                right_fit = np.polyfit(righty, rightx, 2)
                # print(f"Right fit coefficients: {right_fit}")
            except:
                print("Failed to fit right lane polynomial")
                right_fit = None
        
        return left_fit, right_fit, leftx, lefty, rightx, righty
    
    def calculate_confidence(self, fit, x, y, prev_fit=None):
        """Calculate confidence score for lane detection"""
        if fit is None or len(x) == 0:
            return 0.0
        
        # Pixel count score
        pixel_score = min(len(x) / 1000, 1.0)
        
        # Fit residual score
        if len(x) > 0:
            try:
                predicted_x = fit[0]*y**2 + fit[1]*y + fit[2]
                residual = np.mean(np.abs(predicted_x - x))
                residual_score = max(1.0 - residual / 50, 0.0)
            except:
                residual_score = 0.0
        else:
            residual_score = 0.0
        
        # Temporal consistency score
        temporal_score = 1.0
        if prev_fit is not None:
            try:
                coeff_diff = np.sum(np.abs(fit - prev_fit))
                temporal_score = max(1.0 - coeff_diff / 100, 0.0)
            except:
                temporal_score = 0.0
        
        # Combined confidence
        confidence = 0.4 * pixel_score + 0.4 * residual_score + 0.2 * temporal_score
        confidence = max(0.0, min(1.0, confidence))
        
        print(f"Confidence - Pixel: {pixel_score:.3f}, Residual: {residual_score:.3f}, Temporal: {temporal_score:.3f}, Final: {confidence:.3f}")
        return confidence