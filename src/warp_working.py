import cv2
import numpy as np
from typing import Tuple

# Converts camera view to bird's-eye view for easier lane geometry analysis
# Responsibilities:
#     Perspective Transform: Warps image to top-down view using homography
#     Inverse Transform: Converts bird's-eye view back to camera view for display
#     Coordinate Mapping: Defines source and destination points for the transform
#     Debug Visualization: Creates visualizations to verify the transform works
# Key Methods:
#     warp_to_birdeye(): Transforms to top-down view
#     warp_to_camera(): Transforms back to original perspective
#     debug_perspective(): Creates debug images showing the transform
# Why this is important:
#     Makes lane curves appear as straight lines
#     Simplifies lane detection and curve fitting
#     Provides better distance measurements

class PerspectiveTransformer:
    """
    Optimized perspective transformer for reliable lane detection with dynamic adjustment.
    """
    
    def __init__(self, img_size: Tuple[int, int], dynamic_adjust: bool = False):
        """
        Initialize the perspective transformer.

        Args:
            img_size: (width, height) tuple of the image dimensions
            dynamic_adjust: Enable dynamic adjustment of perspective points
        """
        self.img_size = img_size
        self.width, self.height = img_size
        self.dynamic_adjust = dynamic_adjust
        self.M, self.Minv = self._calculate_perspective_matrices()
        self.src_points = None  # Store for dynamic adjustment
        self.dst_points = None
        
    def _calculate_perspective_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate perspective matrices with minimal expansion for lane detection.

        Returns:
            Tuple of (M, Minv) transformation matrices
        """
        # Define conservative source points
        self.src_points = np.float32([
            [self.width * 0.30, self.height * 0.95],  # Bottom-left
            [self.width * 0.45, self.height * 0.65],  # Top-left (adjusted for curves)
            [self.width * 0.55, self.height * 0.65],  # Top-right (adjusted for curves)
            [self.width * 0.70, self.height * 0.95]   # Bottom-right
        ])
        
        # Define destination points with slight expansion
        self.dst_points = np.float32([
            [self.width * 0.30, self.height * 0.90],  # Bottom-left
            [self.width * 0.30, self.height * 0.55],  # Top-left (tighter for focus)
            [self.width * 0.70, self.height * 0.55],  # Top-right
            [self.width * 0.70, self.height * 0.90]   # Bottom-right
        ])
        
        try:
            M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
            
            # Validate transformation
            det_M = np.linalg.det(M)
            if abs(det_M) < 1e-6:
                return self._get_identity_matrices()
            
            # Test transformation ratio
            ratio = self._test_transformation_ratio(M)
            if not 0.85 <= ratio <= 1.15:
                return self._get_identity_matrices()
                
            return M, Minv
            
        except Exception:
            return self._get_identity_matrices()
    
    def _get_identity_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return identity matrices as a fallback."""
        return np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)
    
    def _test_transformation_ratio(self, M: np.ndarray) -> float:
        """Test the pixel preservation ratio of the transformation."""
        try:
            test_img = np.zeros((self.height, self.width), dtype=np.uint8)
            cv2.rectangle(test_img, 
                         (0, int(self.height * 0.6)),
                         (self.width, self.height),
                         255, -1)
            
            warped = cv2.warpPerspective(test_img, M, (self.width, self.height))
            original_pixels = np.sum(test_img > 0)
            warped_pixels = np.sum(warped > 0)
            
            return warped_pixels / original_pixels if original_pixels > 0 else 1.0
        except Exception:
            return 1.0
    
    def update_perspective(self, lane_info: dict = None) -> None:
        """
        Dynamically update perspective points based on lane information.

        Args:
            lane_info: Dictionary with lane histogram peaks (left_peak, right_peak)
        """
        if not self.dynamic_adjust or lane_info is None:
            return
        
        try:
            left_peak = lane_info.get('left_peak', self.width * 0.45)
            right_peak = lane_info.get('right_peak', self.width * 0.55)
            
            # Adjust source points based on lane peaks
            self.src_points[0][0] = max(self.width * 0.20, left_peak - self.width * 0.15)
            self.src_points[1][0] = max(self.width * 0.25, left_peak - self.width * 0.10)
            self.src_points[2][0] = min(self.width * 0.75, right_peak + self.width * 0.10)
            self.src_points[3][0] = min(self.width * 0.80, right_peak + self.width * 0.15)
            
            self.M, self.Minv = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
        except Exception:
            pass
    
    def warp_to_birdeye(self, img: np.ndarray, lane_info: dict = None) -> np.ndarray:
        """
        Transform image to bird's-eye view with controlled expansion.

        Args:
            img: Input image to transform
            lane_info: Optional lane information for dynamic adjustment

        Returns:
            Warped bird's-eye view image
        """
        if not self._validate_input_image(img):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        
        if lane_info:
            self.update_perspective(lane_info)
        
        try:
            return cv2.warpPerspective(
                img, self.M, (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        except Exception:
            return img
    
    def warp_to_camera(self, img: np.ndarray) -> np.ndarray:
        """
        Transform bird's-eye view back to camera view.

        Args:
            img: Bird's-eye view image to transform back

        Returns:
            Original view image
        """
        if not self._validate_input_image(img):
            return np.zeros(self.img_size, dtype=np.uint8)
        
        try:
            return cv2.warpPerspective(
                img, self.Minv, self.img_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        except Exception:
            return np.zeros(self.img_size, dtype=np.uint8)
    
    def _validate_input_image(self, img: np.ndarray) -> bool:
        """Validate input image for transformation."""
        return (img is not None and 
                img.size > 0 and 
                len(img.shape) in [2, 3])
    
    def get_transformation_info(self) -> dict:
        """Get information about the current transformation setup."""
        det = np.linalg.det(self.M) if self.M is not None else 0
        is_identity = np.allclose(self.M, np.eye(3)) if self.M is not None else True
        
        return {
            "image_size": (self.width, self.height),
            "matrix_determinant": det,
            "matrix_type": "identity" if is_identity else "custom",
            "transformer_mode": "dynamic" if self.dynamic_adjust else "conservative"
        }

if __name__ == "__main__":
    transformer = PerspectiveTransformer((640, 480), dynamic_adjust=True)
    info = transformer.get_transformation_info()
    print("Transformation Info:", info)