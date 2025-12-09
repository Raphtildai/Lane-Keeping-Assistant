import cv2
import numpy as np

class OverlayRenderer:
    def __init__(self):
        self.colors = {
            'left_lane': (0, 255, 0),    # Green
            'right_lane': (255, 0, 0),   # Blue
            'uncertain': (128, 128, 128) # Gray
        }
        self.confidence_threshold = 0.6
    
    def draw_lane_polygon(self, image, left_fit, right_fit, Minv, confidence_left, confidence_right):
        """Draw the lane area on the original image, clipped at the horizon/convergence point."""
        if left_fit is None or right_fit is None:
            return image
        
        height, width = image.shape[:2]
        
        # 1. Define the Y-range for plotting (from bottom to top)
        ploty = np.linspace(0, height - 1, height)
        
        # --- CLIPPING POINT DEFINITION ---
        # The clip is based on the top edge of your source trapezoid (Y=0.65*H)
        CLIP_RATIO = 0.65 
        clip_y = int(height * CLIP_RATIO)
        
        # Find the index corresponding to the clip_y coordinate
        clip_index = np.where(ploty >= clip_y)[0][0]
        
        # 2. Calculate the X-points for the full range
        left_fitx_full = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx_full = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
        # 3. CLIP THE COORDINATE ARRAYS
        # We only want points from the clip_index down to the bottom (end of the array)
        ploty_clipped = ploty[clip_index:]
        left_fitx = left_fitx_full[clip_index:]
        right_fitx = right_fitx_full[clip_index:]

        # Create an image to draw the lines on
        warp_zero = np.zeros((height, width), dtype=np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        
        # 4. Create the polygon points using the clipped arrays
        # Point Order: Bottom-left -> Top-left (at clip_y) -> Top-right (at clip_y) -> Bottom-right
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty_clipped]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty_clipped])))]) # Reversed order for proper closing
        
        # The polygon should now close correctly at the clip_y line
        pts = np.hstack((pts_left, pts_right))
        
        # Draw the lane onto the warped blank image
        # Note: Must use np.int32 for cv2.fillPoly
        cv2.fillPoly(color_warp, np.int32(pts), (100, 255, 100)) # Light Green/Yellow
        
        # Warp back to original image space
        newwarp = cv2.warpPerspective(color_warp, Minv, (width, height))
        
        # Combine with original image
        result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
        return result
    
    def draw_lane_lines(self, image, left_fit, right_fit, Minv, confidence_left, confidence_right):
        """Draw left and right lane lines, clipped at the horizon/convergence point."""
        if left_fit is None and right_fit is None:
            return image
        
        height, width = image.shape[:2]
        ploty = np.linspace(0, height - 1, height)

        # --- CLIPPING POINT DEFINITION ---
        CLIP_RATIO = 0.65 
        clip_y = int(height * CLIP_RATIO)
        clip_index = np.where(ploty >= clip_y)[0][0]
        
        # Clip the Y-values
        ploty_clipped = ploty[clip_index:]
        
        # Draw left lane
        if left_fit is not None:
            left_fitx_full = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            left_fitx_clipped = left_fitx_full[clip_index:]

            # Create clipped points array
            left_points = np.array([np.transpose(np.vstack([left_fitx_clipped, ploty_clipped]))], dtype=np.int32)
            
            # FIX SWAP: Left lane should be GREEN (0, 255, 0)
            color = self.colors['left_lane'] if confidence_left > self.confidence_threshold else self.colors['uncertain']
            line_type = cv2.LINE_AA if confidence_left > self.confidence_threshold else cv2.LINE_AA
            
            # 1. Create a blank image
            line_warp = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 2. Draw the line on the warped blank image (bird's eye view)
            cv2.polylines(line_warp, left_points, False, color, thickness=10, lineType=line_type) 
            
            # 3. Warp the line back to original image space
            newwarp = cv2.warpPerspective(line_warp, Minv, (width, height))
            
            # 4. Combine with image (use cv2.addWeighted or simply addition)
            image = cv2.addWeighted(image, 1, newwarp, 1.0, 0)
            
        # Draw right lane (similar logic)
        if right_fit is not None:
            right_fitx_full = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
            right_fitx_clipped = right_fitx_full[clip_index:]

            # Create clipped points array
            right_points = np.array([np.transpose(np.vstack([right_fitx_clipped, ploty_clipped]))], dtype=np.int32)
            
            color = self.colors['right_lane'] if confidence_right > self.confidence_threshold else self.colors['uncertain']
            line_type = cv2.LINE_AA if confidence_right > self.confidence_threshold else cv2.LINE_AA

            # 1. Create a blank image
            line_warp = np.zeros((height, width, 3), dtype=np.uint8)
            
            # 2. Draw the line on the warped blank image (bird's eye view)
            cv2.polylines(line_warp, right_points, False, color, thickness=10, lineType=line_type) 
            
            # 3. Warp the line back to original image space
            newwarp = cv2.warpPerspective(line_warp, Minv, (width, height))
            
            # 4. Combine with image
            image = cv2.addWeighted(image, 1, newwarp, 1.0, 0)
            
        return image
    
    def draw_hud(self, image, left_detected, right_detected, left_conf, right_conf, lat_offset=0.0):
        """Draw heads-up display information with larger font and colored sections"""
        hud_text = [
            ("Road Lane Assist by: Kipchirchir Raphael, LGL7CS", (0, 255, 255)),    # Yellow
            (f"Left: {'YES' if left_detected else 'NO'} | Conf: {left_conf:.2f}", 
            (0, 255, 0) if left_detected else (0, 165, 255)),  # Green if YES, Orange if NO
            (f"Right: {'YES' if right_detected else 'NO'} | Conf: {right_conf:.2f}", 
            (0, 255, 0) if right_detected else (0, 165, 255)),  # Green if YES, Orange if NO
            (f"Lat Offset: {lat_offset:+.2f}m", (255, 255, 0))     # Cyan
        ]
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6        
        thickness = 2
        shadow_offset = (2, 2)
        line_spacing = 40       # Increased spacing for larger text

        for i, (text, color) in enumerate(hud_text):
            y = 40 + i * line_spacing  # Start lower to avoid cutting off top
            
            # Shadow (black)
            cv2.putText(image, text, (10 + shadow_offset[0], y + shadow_offset[1]),
                        font, font_scale, (0, 0, 0), thickness + 1)
            
            # Main text (colored)
            cv2.putText(image, text, (10, y),
                        font, font_scale, color, thickness)
        
        return image
    
    def create_overlay(self, image, left_fit, right_fit, Minv, left_conf, right_conf, lat_offset=0.0):
        """Create complete overlay with lane area, lines, and HUD"""
        # Create base overlay with lane area
        overlay = self.draw_lane_polygon(image.copy(), left_fit, right_fit, Minv, left_conf, right_conf)
        
        # Add lane lines
        overlay = self.draw_lane_lines(overlay, left_fit, right_fit, Minv, left_conf, right_conf)
        
        # Add HUD
        left_detected = left_conf > self.confidence_threshold
        right_detected = right_conf > self.confidence_threshold
        overlay = self.draw_hud(overlay, left_detected, right_detected, left_conf, right_conf, lat_offset)
        
        return overlay