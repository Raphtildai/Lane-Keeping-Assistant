import cv2
import numpy as np
from typing import Tuple

class PerspectiveTransformer:
    """
    Optimized perspective transformer for lane detection with realistic forward-facing camera.
    Converts forward-facing camera view to bird's-eye view for lane geometry analysis.
    """
    
    def __init__(self, img_size: Tuple[int, int], dynamic_adjust: bool = False):
        """
        Initialize the perspective transformer for forward-facing camera.

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
        Calculate perspective matrices for forward-facing camera to bird's-eye view.

        Returns:
            Tuple of (M, Minv) transformation matrices
        """
        # --------------------------------------------------------------------------------
        # FORWARD-FACING CAMERA: Source points (Trapezoid) - ANCHOR ON LANE LINES
        # Using a wide base (10% to 90%) and anchoring the top points 
        # right at the visible convergence point (around Y=0.65).
        self.src_points = np.float32([
            [self.width * 0.10, self.height * 0.98],   # Bottom-left 
            [self.width * 0.46, self.height * 0.65],   # Top-left (Slightly narrower)
            [self.width * 0.54, self.height * 0.65],   # Top-right (Slightly narrower)
            [self.width * 0.90, self.height * 0.98]    # Bottom-right 
        ])
        
        # BIRD'S-EYE VIEW: Destination points (Rectangle) - Projects a long, parallel road
        # Stretching the destination up to Y=0.20 creates a long, parallel, stable box.
        self.dst_points = np.float32([
            [self.width * 0.20, self.height * 0.98], 
            [self.width * 0.20, self.height * 0.20], 
            [self.width * 0.80, self.height * 0.20], 
            [self.width * 0.80, self.height * 0.98]
        ])
        # --------------------------------------------------------------------------------
        
        try:
            M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
            Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
            
            # Validate transformation
            det_M = np.linalg.det(M)
            if abs(det_M) < 1e-6:
                print("⚠️  Using identity matrices due to singular transformation")
                return self._get_identity_matrices()
            
            # Test transformation with realistic lane simulation
            test_ratio = self._test_lane_transformation(M)
            if not 0.7 <= test_ratio <= 1.3:  # More flexible ratio for perspective
                print("⚠️  Transformation test failed, using identity matrices")
                return self._get_identity_matrices()
                
            print("✅ Forward-facing perspective matrices calculated successfully")
            return M, Minv
            
        except Exception as e:
            print(f"❌ Error calculating matrices: {e}")
            return self._get_identity_matrices()
    
    def _get_identity_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return identity matrices as a fallback."""
        return np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)
    
    def _test_lane_transformation(self, M: np.ndarray) -> float:
        """
        Test transformation with realistic lane markings that converge in perspective.
        """
        try:
            # Create test image simulating forward-facing camera view
            test_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw converging lane markings (perspective effect)
            # Left lane - converges toward top-center
            cv2.line(test_img, 
                    (int(self.width * 0.25), self.height - 1),
                    (int(self.width * 0.40), int(self.height * 0.65)),
                    (255, 255, 255), 4)
            
            # Right lane - converges toward top-center  
            cv2.line(test_img,
                    (int(self.width * 0.75), self.height - 1),
                    (int(self.width * 0.60), int(self.height * 0.65)),
                    (255, 255, 255), 4)
            
            # Apply transformation
            warped = cv2.warpPerspective(test_img, M, (self.width, self.height))
            
            # Convert to grayscale for analysis
            test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            
            original_pixels = np.sum(test_gray > 0)
            warped_pixels = np.sum(warped_gray > 0)
            
            ratio = warped_pixels / original_pixels if original_pixels > 0 else 1.0
            print(f"🧪 Lane transformation test ratio: {ratio:.2f}")
            
            return ratio
            
        except Exception as e:
            print(f"❌ Transformation test error: {e}")
            return 1.0
    
    def update_perspective(self, lane_info: dict = None) -> None:
        """
        Dynamically update perspective points based on lane detection results.

        Args:
            lane_info: Dictionary with lane detection information
        """
        if not self.dynamic_adjust or lane_info is None:
            return
        
        try:
            left_peak = lane_info.get('left_peak', self.width * 0.40)
            right_peak = lane_info.get('right_peak', self.width * 0.60)
            confidence = lane_info.get('confidence', 0.5)
            
            # Only adjust if we have high confidence in lane detection
            if confidence > 0.7:
                # Adjust source points to follow lane curvature
                # Keep the perspective trapezoid but shift it based on lane positions
                left_offset = (left_peak - self.width * 0.40) * 0.3
                right_offset = (right_peak - self.width * 0.60) * 0.3
                
                # Update source points while maintaining perspective shape
                self.src_points[0][0] = max(self.width * 0.15, self.width * 0.20 + left_offset)
                self.src_points[1][0] = max(self.width * 0.25, self.width * 0.40 + left_offset)
                self.src_points[2][0] = min(self.width * 0.85, self.width * 0.60 + right_offset)
                self.src_points[3][0] = min(self.width * 0.90, self.width * 0.80 + right_offset)
                
                # Recalculate transformation matrices
                self.M = cv2.getPerspectiveTransform(self.src_points, self.dst_points)
                self.Minv = cv2.getPerspectiveTransform(self.dst_points, self.src_points)
                
                print(f"🔄 Adjusted perspective: left_peak={left_peak:.1f}, right_peak={right_peak:.1f}")
                
        except Exception as e:
            print(f"❌ Error updating perspective: {e}")
    
    def warp_to_birdeye(self, img: np.ndarray, lane_info: dict = None) -> np.ndarray:
        """
        Transform forward-facing camera view to bird's-eye view.

        Args:
            img: Input image from forward-facing camera
            lane_info: Optional lane information for dynamic adjustment

        Returns:
            Warped bird's-eye view image
        """
        if not self._validate_input_image(img):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if img.shape[:2] != (self.height, self.width):
            img = cv2.resize(img, (self.width, self.height))
        
        if lane_info and self.dynamic_adjust:
            self.update_perspective(lane_info)
        
        try:
            warped = cv2.warpPerspective(
                img, self.M, (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Log transformation quality
            self._log_transformation_quality(img, warped)
            return warped
            
        except Exception as e:
            print(f"❌ Error in warp_to_birdeye: {e}")
            return img
    
    def warp_to_camera(self, img: np.ndarray) -> np.ndarray:
        """
        Transform bird's-eye view back to forward-facing camera view.

        Args:
            img: Bird's-eye view image to transform back

        Returns:
            Forward-facing camera view image
        """
        if not self._validate_input_image(img):
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        try:
            return cv2.warpPerspective(
                img, self.Minv, self.img_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        except Exception as e:
            print(f"❌ Error in warp_to_camera: {e}")
            return np.zeros(self.img_size, dtype=np.uint8)
    
    def _validate_input_image(self, img: np.ndarray) -> bool:
        """Validate input image for transformation."""
        return (img is not None and 
                img.size > 0 and 
                len(img.shape) in [2, 3])
    
    def _log_transformation_quality(self, original: np.ndarray, warped: np.ndarray) -> None:
        """Log the quality of the perspective transformation."""
        try:
            if len(original.shape) == 3:
                original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            else:
                original_gray = original
                warped_gray = warped
            
            original_pixels = np.sum(original_gray > 0)
            warped_pixels = np.sum(warped_gray > 0)
            
            if original_pixels > 0:
                ratio = warped_pixels / original_pixels
                # print(f"📊 Transformation: {original_pixels} → {warped_pixels} pixels ({ratio:.1%})")
        except:
            pass
    
    def visualize_perspective_areas(self, img: np.ndarray) -> np.ndarray:
        """
        Create visualization showing the perspective transformation areas.
        
        Args:
            img: Input image to draw on
            
        Returns:
            Image with perspective areas visualized
        """
        if img is None or not self._validate_input_image(img):
            return img
            
        vis_img = img.copy()
        
        # Draw source trapezoid (forward-facing camera view)
        if self.src_points is not None:
            src_int = self.src_points.astype(int)
            cv2.polylines(vis_img, [src_int], True, (0, 255, 0), 2)
            for i, (x, y) in enumerate(src_int):
                cv2.circle(vis_img, (x, y), 5, (0, 255, 0), -1)
                cv2.putText(vis_img, f'S{i}', (x-10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Draw destination rectangle (bird's-eye view)
        if self.dst_points is not None:
            # Transform destination points back to camera view for visualization
            dst_camera_view = cv2.perspectiveTransform(
                self.dst_points.reshape(-1, 1, 2), self.Minv
            ).reshape(-1, 2).astype(int)
            
            cv2.polylines(vis_img, [dst_camera_view], True, (255, 0, 0), 2)
            for i, (x, y) in enumerate(dst_camera_view):
                cv2.circle(vis_img, (x, y), 5, (255, 0, 0), -1)
                cv2.putText(vis_img, f'D{i}', (x-10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Add legend
        cv2.putText(vis_img, 'Green: Camera View (Source)', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(vis_img, 'Blue: Bird\'s-Eye View (Dest)', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        return vis_img
    
    def get_transformation_info(self) -> dict:
        """Get information about the current transformation setup."""
        det = np.linalg.det(self.M) if self.M is not None else 0
        is_identity = np.allclose(self.M, np.eye(3)) if self.M is not None else True
        
        return {
            "image_size": (self.width, self.height),
            "matrix_determinant": det,
            "matrix_type": "identity" if is_identity else "custom",
            "transformer_mode": "dynamic" if self.dynamic_adjust else "static",
            "camera_type": "forward_facing"
        }


# Demonstration and testing
if __name__ == "__main__":
    # Test with different image sizes
    transformer = PerspectiveTransformer((640, 480), dynamic_adjust=True)
    info = transformer.get_transformation_info()
    print("Transformation Info:", info)
    
    # Create a realistic test image simulating forward-facing camera
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw road with perspective (converging lanes)
    road_color = (100, 100, 100)
    cv2.rectangle(test_img, (0, 240), (640, 480), road_color, -1)
    
    # Left lane marking (converging)
    cv2.line(test_img, (200, 480), (280, 300), (255, 255, 255), 4)
    # Right lane marking (converging)
    cv2.line(test_img, (440, 480), (360, 300), (255, 255, 255), 4)
    
    # Create visualization
    vis_img = transformer.visualize_perspective_areas(test_img)
    
    # Apply transformation
    birdseye = transformer.warp_to_birdeye(test_img)
    
    # Display results
    combined = np.hstack([vis_img, birdseye])
    cv2.putText(combined, 'Forward-Facing Camera View', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, 'Bird\'s-Eye View', (650, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Forward-Facing Perspective Transformation', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()