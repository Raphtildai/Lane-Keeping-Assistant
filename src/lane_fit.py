import numpy as np
import cv2

class LaneDetector:
    def __init__(self, n_windows=9, margin=100, minpix=50):
        self.n_windows = n_windows
        self.margin = margin
        self.minpix = minpix
        self.ym_per_pix = 30 / 720  # meters per pixel in y dimension
        self.xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
        # Initialize placeholders for last successful fits for fallback
        self.last_left_fit = None
        self.last_right_fit = None
        
    def find_lane_base(self, binary_warped):
        """Find starting position for left and right lanes using histogram"""
        if binary_warped is None or binary_warped.size == 0:
            return None, None
            
        # Take a histogram of the bottom half of the image
        # CRITICAL STABILITY TWEAK: Use only the bottom quarter for more robust base detection
        y_cutoff = int(binary_warped.shape[0] * 0.75) 
        histogram = np.sum(binary_warped[y_cutoff:, :], axis=0) 
        
        if np.max(histogram) == 0:
            print("No lane pixels found in histogram")
            return None, None
            
        midpoint = histogram.shape[0] // 2
        
        # Find the peak of the left and right halves of the histogram
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
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
        
        # 🛑 CRITICAL FIX: Ensure lane_inds are usable numpy arrays
        if not isinstance(left_lane_inds, np.ndarray):
            left_lane_inds = np.array(left_lane_inds, dtype=np.int64)
        if not isinstance(right_lane_inds, np.ndarray):
            right_lane_inds = np.array(right_lane_inds, dtype=np.int64)

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        left_fit = self.last_left_fit # Use last successful fit as a base
        right_fit = self.last_right_fit
        
        # Initialize pixel lists to empty arrays
        leftx = np.array([])
        lefty = np.array([]) 
        rightx = np.array([])
        righty = np.array([])

        # Check for empty index arrays BEFORE indexing (PREVENTS 'slice' ERROR)
        if left_lane_inds.size > 0:
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds] 
            try:
                left_fit = np.polyfit(lefty, leftx, 2)
                self.last_left_fit = left_fit # Update last good fit
            except Exception as e:
                print(f"Failed to fit left lane polynomial: {e}")
                
        if right_lane_inds.size > 0:
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            try:
                right_fit = np.polyfit(righty, rightx, 2)
                self.last_right_fit = right_fit # Update last good fit
            except Exception as e:
                print(f"Failed to fit right lane polynomial: {e}")
        
        return left_fit, right_fit, leftx, lefty, rightx, righty
    
    def calculate_confidence(self, fit, x, y, prev_fit=None):
        """Calculate confidence score for lane detection"""
        if fit is None or len(x) == 0:
            return 0.0
        
        # Pixel count score
        pixel_score = min(len(x) / 1000, 1.0) # Assume 1000 pixels is "perfect"
        
        # Fit residual score
        if len(x) > 0:
            try:
                # Use np.polyval for safer evaluation
                predicted_x = np.polyval(fit, y)
                residual = np.mean(np.abs(predicted_x - x))
                # Adjusting divisor for more reasonable scoring
                residual_score = max(1.0 - residual / 25, 0.0) 
            except:
                residual_score = 0.0
        else:
            residual_score = 0.0
        
        # Temporal consistency score
        temporal_score = 1.0
        if prev_fit is not None and prev_fit.size == fit.size: # Check size before subtraction
            try:
                # Increase sensitivity to changes by lowering the divisor
                coeff_diff = np.sum(np.abs(fit - prev_fit))
                temporal_score = max(1.0 - coeff_diff / 25, 0.0) 
            except:
                temporal_score = 0.0
        
        # Combined confidence (Adjusting weights for more reliance on Residual and Temporal)
        confidence = 0.2 * pixel_score + 0.5 * residual_score + 0.3 * temporal_score
        confidence = max(0.0, min(1.0, confidence))
        
        print(f"Confidence - Pixel: {pixel_score:.3f}, Residual: {residual_score:.3f}, Temporal: {temporal_score:.3f}, Final: {confidence:.3f}")
        return confidence
    
    def draw_detection(self, 
                       birdseye_img, 
                       leftx, lefty, rightx, righty, 
                       left_fit, right_fit):
        """
        Draws the detected points and the fitted polynomial lines 
        onto the bird's-eye view for visualization.
        """
        # Convert the single-channel binary image to a 3-channel color image
        out_img = cv2.cvtColor(birdseye_img * 255, cv2.COLOR_GRAY2BGR)
        
        # 1. Draw the detected pixels (optional: can be memory intensive)
        # out_img[lefty, leftx] = [255, 0, 0] # Blue for left pixels
        # out_img[righty, rightx] = [0, 0, 255] # Red for right pixels

        # 2. Draw the fitted polynomial lines
        ploty = np.linspace(0, birdseye_img.shape[0] - 1, birdseye_img.shape[0])
        
        if left_fit is not None:
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            for i in range(len(ploty)):
                x = int(left_fitx[i])
                y = int(ploty[i])
                if 0 <= x < birdseye_img.shape[1]:
                    cv2.circle(out_img, (x, y), 3, (0, 255, 255), -1) # Yellow for smoothed left fit

        if right_fit is not None:
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            for i in range(len(ploty)):
                x = int(right_fitx[i])
                y = int(ploty[i])
                if 0 <= x < birdseye_img.shape[1]:
                    cv2.circle(out_img, (x, y), 3, (0, 255, 255), -1) # Yellow for smoothed right fit
        
        return out_img