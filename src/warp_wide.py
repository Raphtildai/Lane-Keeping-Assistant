import cv2
import numpy as np
from typing import Tuple, Optional, Union

class PerspectiveTransformer:
    """
    A robust perspective transformer for lane detection applications.
    Handles bird's-eye view transformation from forward-facing camera.
    """
    
    def __init__(self, img_size: Tuple[int, int], camera_angle: float = 25.0):
        """
        Initialize the perspective transformer for forward-facing camera.
        
        Args:
            img_size: (width, height) tuple of the image dimensions
            camera_angle: Approximate camera angle in degrees from horizontal
        """
        self.img_size = img_size
        self.width, self.height = img_size
        self.camera_angle = camera_angle
        self.M, self.Minv = self._calculate_forward_facing_perspective_matrices()
        
        # Validate the transformation matrices on initialization
        self._validate_transformation_matrices()
    
    def _calculate_forward_facing_perspective_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate perspective transformation matrices for forward-facing camera.
        This simulates looking at the road ahead from a vehicle-mounted camera.
        """
        print(f"🔄 Initializing forward-facing perspective transform for {self.width}x{self.height} image")
        print(f"📷 Simulating camera angle: {self.camera_angle}° from horizontal")
        
        try:
            # FORWARD-FACING CAMERA: Points that simulate a camera looking ahead at the road
            # The trapezoid represents the road stretching into the distance
            
            # Source points - trapezoid that gets narrower at the top (vanishing point)
            src = np.float32([
                [self.width * 0.15, self.height * 0.95],  # Bottom-left (near the car)
                [self.width * 0.45, self.height * 0.65],  # Top-left (further away)
                [self.width * 0.55, self.height * 0.65],  # Top-right (further away)
                [self.width * 0.85, self.height * 0.95]   # Bottom-right (near the car)
            ])
            
            # Destination points - rectangle for bird's-eye view
            # This transforms the trapezoidal road view to a top-down rectangular view
            dst = np.float32([
                [self.width * 0.20, self.height],  # Bottom-left
                [self.width * 0.20, self.height * 0.2],  # Top-left (higher up = further away)
                [self.width * 0.80, self.height * 0.2],  # Top-right (higher up = further away)
                [self.width * 0.80, self.height]   # Bottom-right
            ])
            
            # Adjust based on camera angle for more realism
            if self.camera_angle > 30:  # Higher camera angle
                src = np.float32([
                    [self.width * 0.10, self.height * 0.90],
                    [self.width * 0.40, self.height * 0.55],
                    [self.width * 0.60, self.height * 0.55],
                    [self.width * 0.90, self.height * 0.90]
                ])
            elif self.camera_angle < 20:  # Lower camera angle
                src = np.float32([
                    [self.width * 0.20, self.height * 0.98],
                    [self.width * 0.45, self.height * 0.75],
                    [self.width * 0.55, self.height * 0.75],
                    [self.width * 0.80, self.height * 0.98]
                ])
            
            self._log_points("Source points (camera view)", src)
            self._log_points("Destination points (bird's-eye view)", dst)
            
            # Validate point configuration before creating matrices
            if not self._validate_point_configuration(src, dst):
                print("🔄 Trying alternative point configuration...")
                return self._calculate_fallback_matrices()
            
            M = cv2.getPerspectiveTransform(src, dst)
            Minv = cv2.getPerspectiveTransform(dst, src)
            
            # Test the transformation with a simple grid
            test_result = self._test_transformation(M)
            if not test_result:
                print("🔄 Transformation test failed, using fallback...")
                return self._calculate_fallback_matrices()
            
            print("✅ Forward-facing perspective matrices calculated successfully")
            return M, Minv
            
        except Exception as e:
            print(f"❌ Error calculating perspective matrices: {e}")
            return self._calculate_fallback_matrices()
    
    def _calculate_fallback_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate fallback matrices for forward-facing camera."""
        print("🔄 Calculating fallback forward-facing perspective matrices...")
        
        # Conservative forward-facing points
        src = np.float32([
            [self.width * 0.2, self.height],      # Bottom-left
            [self.width * 0.4, self.height * 0.6], # Top-left
            [self.width * 0.6, self.height * 0.6], # Top-right
            [self.width * 0.8, self.height]       # Bottom-right
        ])
        
        dst = np.float32([
            [self.width * 0.3, self.height],      # Bottom-left
            [self.width * 0.3, self.height * 0.3], # Top-left
            [self.width * 0.7, self.height * 0.3], # Top-right
            [self.width * 0.7, self.height]       # Bottom-right
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        print("✅ Fallback matrices calculated")
        return M, Minv
    
    def _test_transformation(self, M: np.ndarray) -> bool:
        """Test if the transformation matrix works correctly."""
        try:
            # Create a test image that simulates lane markings on a road
            test_img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Draw lane markings (simulating a forward-facing camera view)
            # Left lane marking
            cv2.line(test_img, 
                    (int(self.width * 0.3), int(self.height * 0.9)),
                    (int(self.width * 0.4), int(self.height * 0.6)),
                    (255, 255, 255), 3)
            # Right lane marking
            cv2.line(test_img,
                    (int(self.width * 0.7), int(self.height * 0.9)),
                    (int(self.width * 0.6), int(self.height * 0.6)),
                    (255, 255, 255), 3)
            
            # Apply transformation
            warped = cv2.warpPerspective(test_img, M, (self.width, self.height))
            
            # Check if we got reasonable output
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
            warped_pixels = np.sum(warped_gray > 0)
            original_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)
            original_pixels = np.sum(original_gray > 0)
            
            if warped_pixels == 0 and original_pixels > 0:
                print("❌ Transformation test failed: No output pixels")
                return False
            
            preservation_ratio = warped_pixels / original_pixels if original_pixels > 0 else 0
            print(f"🧪 Transformation test: {preservation_ratio:.1%} preservation")
            
            # Visualize the test (optional, for debugging)
            if False:  # Set to True to see test visualization
                combined = np.hstack([test_img, warped])
                cv2.imshow('Transformation Test', combined)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            return preservation_ratio > 0.1  # At least 10% preservation
            
        except Exception as e:
            print(f"❌ Transformation test error: {e}")
            return False
    
    def _validate_transformation_matrices(self) -> bool:
        """Validate that transformation matrices are properly formed."""
        if self.M is None or self.Minv is None:
            print("❌ Transformation matrices are None")
            return False
        
        if self.M.shape != (3, 3) or self.Minv.shape != (3, 3):
            print("❌ Transformation matrices have incorrect shape")
            return False
        
        # Check if matrices are identity (fallback)
        if np.allclose(self.M, np.eye(3)) and np.allclose(self.Minv, np.eye(3)):
            print("⚠️  Using identity matrices (minimal transformation)")
            return True
        
        # Check if matrices are invertible
        try:
            det_M = np.linalg.det(self.M)
            det_Minv = np.linalg.det(self.Minv)
            
            if abs(det_M) < 1e-6 or abs(det_Minv) < 1e-6:
                print("❌ Transformation matrix is nearly singular")
                return False
                
            print("✅ Transformation matrices validated successfully")
            return True
            
        except Exception as e:
            print(f"❌ Matrix validation error: {e}")
            return False
    
    def _validate_point_configuration(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """Validate that point configurations form valid quadrilaterals."""
        try:
            # Check if points are within image bounds
            for i, (x, y) in enumerate(src):
                if x < 0 or x > self.width or y < 0 or y > self.height:
                    print(f"❌ Source point {i} out of bounds: ({x:.1f}, {y:.1f})")
                    return False
            
            # Check area
            src_area = cv2.contourArea(src)
            dst_area = cv2.contourArea(dst)
            
            print(f"📐 Source area: {src_area:.0f}, Destination area: {dst_area:.0f}")
            
            if src_area < 1000:
                print("❌ Source area too small")
                return False
            
            # Check for convex quadrilateral
            src_hull = cv2.convexHull(src.reshape(-1, 1, 2))
            if len(src_hull) != 4:
                print("❌ Source points do not form a convex quadrilateral")
                return False
                
            return True
            
        except Exception as e:
            print(f"❌ Point configuration validation failed: {e}")
            return False
    
    def _log_points(self, title: str, points: np.ndarray) -> None:
        """Log point coordinates in a formatted way."""
        print(f"{title}:")
        for i, (x, y) in enumerate(points):
            print(f"  Point {i}: ({x:6.1f}, {y:6.1f})")
    
    def visualize_transformation_areas(self, img: np.ndarray) -> np.ndarray:
        """
        Visualize the source and destination transformation areas on the image.
        
        Args:
            img: Input image to draw on
            
        Returns:
            Image with transformation areas visualized
        """
        if img is None:
            return None
            
        vis_img = img.copy()
        
        # Draw source area (forward-facing camera view)
        src = np.float32([
            [self.width * 0.15, self.height * 0.95],
            [self.width * 0.45, self.height * 0.65],
            [self.width * 0.55, self.height * 0.65],
            [self.width * 0.85, self.height * 0.95]
        ])
        
        # Draw source trapezoid
        src_points = src.astype(int)
        cv2.polylines(vis_img, [src_points], True, (0, 255, 0), 2)
        for i, (x, y) in enumerate(src_points):
            cv2.circle(vis_img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(vis_img, f'S{i}', (x-10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add informational text
        cv2.putText(vis_img, 'Forward-Facing Camera View', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(vis_img, 'Green: Source (Camera View)', (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return vis_img
    
    def warp_to_birdeye(self, img: np.ndarray) -> np.ndarray:
        """
        Transform forward-facing camera view to bird's-eye view.
        
        Args:
            img: Input image from forward-facing camera
            
        Returns:
            Warped bird's-eye view image
        """
        if not self._validate_input_image(img):
            return np.zeros((self.height, self.width, 3) if len(img.shape) == 3 else (self.height, self.width), 
                          dtype=img.dtype if img is not None else np.uint8)
        
        try:
            # Ensure image is the correct size
            if img.shape[1] != self.width or img.shape[0] != self.height:
                img = cv2.resize(img, (self.width, self.height))
            
            # Apply perspective transformation
            warped = cv2.warpPerspective(
                img, self.M, (self.width, self.height), 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
            
            # Analyze transformation quality
            quality_ok = self._analyze_transformation_quality(img, warped)
            
            if not quality_ok:
                print("🔄 Low transformation quality, applying corrective measures...")
                return self._apply_corrective_measures(img, warped)
            
            return warped
            
        except Exception as e:
            print(f"❌ Error in warp_to_birdeye: {e}")
            return self._get_fallback_result(img)
    
    def _apply_corrective_measures(self, original: np.ndarray, warped: np.ndarray) -> np.ndarray:
        """Apply corrective measures when transformation quality is poor."""
        warped_pixels = np.sum(warped > 0) if len(warped.shape) == 2 else np.sum(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0)
        
        if warped_pixels == 0:
            print("🔄 No pixels in warped image, trying alternative approaches...")
            
            # Try with different border mode
            try:
                warped_alternative = cv2.warpPerspective(
                    original, self.M, (self.width, self.height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                alt_pixels = np.sum(warped_alternative > 0) if len(warped_alternative.shape) == 2 else np.sum(cv2.cvtColor(warped_alternative, cv2.COLOR_BGR2GRAY) > 0)
                if alt_pixels > 0:
                    print("✅ Alternative border mode successful")
                    return warped_alternative
            except:
                pass
        
        # If all else fails, return a minimally processed version
        print("⚠️  Using edge-based fallback")
        if len(original.shape) == 3:
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        else:
            gray = original
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR) if len(original.shape) == 3 else edges
    
    def warp_to_camera(self, img: np.ndarray) -> np.ndarray:
        """
        Transform bird's-eye view back to forward-facing camera view.
        
        Args:
            img: Bird's-eye view image to transform back
            
        Returns:
            Forward-facing camera view image
        """
        if not self._validate_input_image(img):
            return np.zeros((self.height, self.width, 3) if len(img.shape) == 3 else (self.height, self.width), 
                          dtype=img.dtype if img is not None else np.uint8)
        
        try:
            return cv2.warpPerspective(
                img, self.Minv, self.img_size, 
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )
        except Exception as e:
            print(f"❌ Error in warp_to_camera: {e}")
            return np.zeros((self.height, self.width, 3) if len(img.shape) == 3 else (self.height, self.width), 
                          dtype=img.dtype if img is not None else np.uint8)
    
    def _validate_input_image(self, img: np.ndarray) -> bool:
        """Validate input image for transformation."""
        if img is None:
            print("❌ Input image is None")
            return False
        
        if img.size == 0:
            print("❌ Input image is empty")
            return False
        
        if len(img.shape) not in [2, 3]:
            print("❌ Input image has invalid dimensions")
            return False
        
        return True
    
    def _analyze_transformation_quality(self, original: np.ndarray, warped: np.ndarray) -> bool:
        """Analyze and log the quality of the perspective transformation."""
        if len(original.shape) == 3:
            original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if len(warped.shape) == 3 else warped
        else:
            original_gray = original
            warped_gray = warped
            
        original_pixels = np.sum(original_gray > 0) if original is not None else 0
        warped_pixels = np.sum(warped_gray > 0) if warped is not None else 0
        
        print(f"🔄 Warp: {original_pixels} → {warped_pixels} pixels")
        
        if original_pixels > 0:
            preservation_ratio = warped_pixels / original_pixels
            print(f"📊 Pixel preservation: {preservation_ratio:.2%}")
            
            if warped_pixels == 0:
                print("❌ CRITICAL: No pixels preserved in transformation")
                return False
            elif preservation_ratio < 0.1:
                print("⚠️  LOW: Less than 10% pixels preserved")
                return False
            elif preservation_ratio > 0.3:
                print("✅ GOOD: Reasonable pixel preservation")
                return True
            else:
                print("⚠️  MARGINAL: Low but acceptable preservation")
                return True
        
        return warped_pixels > 0
    
    def _get_fallback_result(self, img: np.ndarray) -> np.ndarray:
        """Get fallback result when transformation fails."""
        print("🔄 Attempting fallback strategies...")
        
        # Strategy 1: Try nearest neighbor interpolation
        try:
            warped = cv2.warpPerspective(
                img, self.M, (self.width, self.height), 
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_REPLICATE
            )
            warped_pixels = np.sum(warped > 0) if len(warped.shape) == 2 else np.sum(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 0)
            if warped_pixels > 0:
                print("✅ Fallback 1 (NEAREST) successful")
                return warped
        except:
            pass
        
        # Strategy 2: Return original image with warning
        print("⚠️  All fallbacks failed, returning original image")
        return img if img is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def get_transformation_info(self) -> dict:
        """Get information about the current transformation setup."""
        det = np.linalg.det(self.M) if self.M is not None else 0
        return {
            "image_size": (self.width, self.height),
            "camera_angle": self.camera_angle,
            "matrix_determinant": det,
            "matrix_type": "identity" if np.allclose(self.M, np.eye(3)) else "custom",
            "transformation_type": "forward_facing_to_birdeye"
        }


# Example usage and demonstration
if __name__ == "__main__":
    # Test the forward-facing transformer
    print("🚗 Testing Forward-Facing Camera Perspective Transformer")
    transformer = PerspectiveTransformer((640, 480), camera_angle=25.0)
    info = transformer.get_transformation_info()
    print("Transformation Info:", info)
    
    # Create a test image that simulates a forward-facing camera view
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw road markings (perspective lines converging at the top)
    # Left lane marking
    cv2.line(test_img, (200, 450), (250, 300), (255, 255, 255), 3)
    # Right lane marking  
    cv2.line(test_img, (440, 450), (390, 300), (255, 255, 255), 3)
    # Center dashed line
    for i in range(300, 450, 30):
        cv2.line(test_img, (320, i), (320, i+15), (255, 255, 255), 2)
    
    # Visualize transformation areas
    vis_img = transformer.visualize_transformation_areas(test_img)
    
    # Apply transformation
    birdseye = transformer.warp_to_birdeye(test_img)
    
    # Display results
    combined = np.hstack([vis_img, birdseye])
    cv2.putText(combined, 'Forward-Facing View', (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(combined, 'Bird\'s-Eye View', (650, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.imshow('Forward-Facing Camera Perspective Transformation', combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()