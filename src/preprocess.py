import cv2
import numpy as np

# Transforms raw video frames into binary images highlighting lane markings
# Responsibilities:
#     Color Space Conversion: Converts BGR to HLS, HSV, LAB for better lane detection
#     Thresholding: Applies multiple thresholds to detect white/yellow lane markings
#     Edge Detection: Uses Sobel operators to find vertical edges (lane boundaries)
#     Noise Reduction: Applies morphological operations to clean up the binary image
#     Region of Interest: Focuses processing on the road area (ignores sky, etc.)
# Key Methods:
#     combine_thresholds(): Main method that combines all detection techniques
#     adaptive_white_detection(): Specifically targets white lane markings
#     edge_detection(): Finds vertical edges using Sobel operators
#     region_of_interest(): Masks non-road areas

class LanePreprocessor:
    def __init__(self):
        # Conservative thresholds for lane detection
        self.kernel_size = 3
        
    def adaptive_white_detection(self, img):
        """Adaptive white detection using multiple color spaces"""
        # Convert to different color spaces
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        
        # Method 1: HSV - White has low saturation, high value
        white_hsv = np.zeros_like(hsv[:,:,0])
        white_hsv[(hsv[:,:,1] <= 30) & (hsv[:,:,2] >= 200)] = 1
        
        # Method 2: LAB - L channel for brightness
        white_lab = np.zeros_like(lab[:,:,0])
        white_lab[(lab[:,:,0] >= 220)] = 1
        
        # Method 3: HLS - High lightness
        white_hls = np.zeros_like(hls[:,:,0])
        white_hls[(hls[:,:,1] >= 220)] = 1
        
        # Combine methods with AND logic (more conservative)
        white_combined = np.zeros_like(white_hsv)
        white_combined[(white_hsv == 1) & (white_lab == 1)] = 1
        
        # Also include pixels that are very bright in both LAB and HLS
        very_bright = np.zeros_like(white_hsv)
        very_bright[(white_lab == 1) & (white_hls == 1)] = 1
        
        final_white = np.zeros_like(white_hsv)
        final_white[(white_combined == 1) | (very_bright == 1)] = 1
        
        return final_white
    
    def edge_detection(self, img):
        """Edge detection focusing on vertical edges (lane markings)"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Sobel x derivative (vertical edges)
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        abs_sobelx = np.absolute(sobelx)
        
        # Scale to 8-bit
        if np.max(abs_sobelx) > 0:
            scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))
        else:
            scaled_sobel = np.zeros_like(abs_sobelx, dtype=np.uint8)
        
        # Threshold for strong edges only
        edge_binary = np.zeros_like(scaled_sobel)
        edge_binary[scaled_sobel >= 50] = 1
        
        return edge_binary
    
    def morphological_cleanup(self, binary_img):
        """Clean up binary image using morphological operations"""
        # Remove small noise
        kernel_open = np.ones((3, 3), np.uint8)
        cleaned = cv2.morphologyEx(binary_img.astype(np.uint8), cv2.MORPH_OPEN, kernel_open)
        
        # Connect nearby lane markings
        kernel_close = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_close)
        
        return cleaned
    
    def region_of_interest(self, img):
        """Apply region of interest mask (focus on road area)"""
        mask = np.zeros_like(img)
        height, width = img.shape
        
        # Define a trapezoid covering the road area
        vertices = np.array([[
            (width * 0.1, height),          # Bottom-left
            (width * 0.45, height * 0.6),   # Top-left
            (width * 0.55, height * 0.6),   # Top-right
            (width * 0.9, height)           # Bottom-right
        ]], dtype=np.int32)
        
        cv2.fillPoly(mask, vertices, 1)
        masked_img = cv2.bitwise_and(img, mask)
        
        return masked_img
    
    def combine_thresholds(self, img):
        """Main thresholding function - very conservative approach"""
        # print("Starting conservative thresholding...")
        
        # Step 1: Adaptive white detection
        white_binary = self.adaptive_white_detection(img)
        white_pixels = np.sum(white_binary)
        # print(f"White detection: {white_pixels} pixels")
        
        # If we're detecting too much, be more aggressive
        total_pixels = img.shape[0] * img.shape[1]
        if white_pixels > total_pixels * 0.05:  # More than 5% of image
            # print("Too many white pixels detected, applying aggressive filtering")
            
            # Method 1: Use only the brightest pixels in LAB space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            white_binary = np.zeros_like(l_channel)
            white_binary[l_channel >= 240] = 1  # Only very bright pixels
            
            # Method 2: Combine with saturation check
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            saturation_mask = hsv[:,:,1] <= 20  # Very low saturation
            white_binary = white_binary & saturation_mask
            
            # print(f"After aggressive filtering: {np.sum(white_binary)} pixels")
        
        # Step 2: Edge detection
        edge_binary = self.edge_detection(img)
        # print(f"Edge detection: {np.sum(edge_binary)} pixels")
        
        # Step 3: Combine methods conservatively
        combined = np.zeros_like(white_binary)
        
        # Use white detection as primary method
        combined[white_binary == 1] = 1
        
        # Add edges only if they overlap with white areas
        edges_near_white = (edge_binary == 1) 
        combined[edges_near_white] = 1
        
        # print(f"After combination: {np.sum(combined)} pixels")
        
        # Step 4: Apply region of interest
        combined = self.region_of_interest(combined)
        # print(f"After ROI: {np.sum(combined)} pixels")
        
        # Step 5: Morphological cleanup
        combined = self.morphological_cleanup(combined)
        # print(f"After cleanup: {np.sum(combined)} pixels")
        
        # Final check: if still too many pixels, use most conservative approach
        final_pixels = np.sum(combined)
        if final_pixels > total_pixels * 0.02:  # More than 2% of image
            # print("Still too many pixels, using ultra-conservative approach")
            
            # Only keep the very brightest clusters
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel = lab[:,:,0]
            
            # Threshold for very bright areas only
            _, ultra_binary = cv2.threshold(l_channel, 250, 1, cv2.THRESH_BINARY)
            
            # Apply ROI and cleanup
            ultra_binary = self.region_of_interest(ultra_binary)
            ultra_binary = self.morphological_cleanup(ultra_binary)
            
            combined = ultra_binary
            # print(f"Ultra-conservative result: {np.sum(combined)} pixels")
        
        percentage = (np.sum(combined) / total_pixels) * 100
        # print(f"Final result: {np.sum(combined)} pixels ({percentage:.2f}% of image)")
        
        return combined
    
    def debug_thresholds(self, img):
        """Debug function to show intermediate steps"""
        white_binary = self.adaptive_white_detection(img)
        edge_binary = self.edge_detection(img)
        combined = self.combine_thresholds(img)
        
        # Create debug visualization
        height, width = img.shape[:2]
        debug_img = np.zeros((height * 2, width * 2, 3), dtype=np.uint8)
        
        # Original image
        debug_img[0:height, 0:width] = img
        
        # White detection
        debug_img[0:height, width:width*2] = cv2.cvtColor(white_binary * 255, cv2.COLOR_GRAY2BGR)
        
        # Edge detection
        debug_img[height:height*2, 0:width] = cv2.cvtColor(edge_binary * 255, cv2.COLOR_GRAY2BGR)
        
        # Combined result
        debug_img[height:height*2, width:width*2] = cv2.cvtColor(combined * 255, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(debug_img, "Original", (10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, "White Detection", (width + 10, 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, "Edge Detection", (10, height + 30), font, 1, (255, 255, 255), 2)
        cv2.putText(debug_img, "Combined", (width + 10, height + 30), font, 1, (255, 255, 255), 2)
        
        return debug_img