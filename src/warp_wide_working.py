import cv2
import numpy as np
from typing import Tuple, Optional, Union

class PerspectiveTransformer:
    """
    A robust perspective transformer for lane detection applications.
    Handles bird's-eye view transformation with comprehensive error handling.
    """
    
    def __init__(self, img_size: Tuple[int, int]):
        """
        Initialize the perspective transformer.
        
        Args:
            img_size: (width, height) tuple of the image dimensions
        """
        self.img_size = img_size
        self.width, self.height = img_size
        self.M, self.Minv = self._calculate_optimized_perspective_matrices()
        
        # Validate the transformation matrices on initialization
        self._validate_transformation_matrices()
    
    def _calculate_optimized_perspective_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate perspective transformation matrices with robust error handling.
        
        Returns:
            Tuple of (M, Minv) transformation matrices
        """
        print(f"🔄 Initializing perspective transform for {self.width}x{self.height} image")
        
        try:
            # FIXED: More conservative source points that are guaranteed to be within image bounds
            src = np.float32([
                [self.width * 0.10, self.height * 0.95],  # Bottom-left (well within bounds)
                [self.width * 0.40, self.height * 0.70],  # Top-left
                [self.width * 0.60, self.height * 0.70],  # Top-right
                [self.width * 0.90, self.height * 0.95]   # Bottom-right (well within bounds)
            ])
            
            # Destination points for bird's-eye view (wider to capture more content)
            dst = np.float32([
                [self.width * 0.10, self.height],  # Bottom-left (extend to bottom)
                [self.width * 0.10, 0],           # Top-left (extend to top)
                [self.width * 0.90, 0],           # Top-right (extend to top)
                [self.width * 0.90, self.height]  # Bottom-right (extend to bottom)
            ])
            
            self._log_points("Source points", src)
            self._log_points("Destination points", dst)
            
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
            
            print("✅ Perspective matrices calculated successfully")
            return M, Minv
            
        except Exception as e:
            print(f"❌ Error calculating perspective matrices: {e}")
            return self._calculate_fallback_matrices()
    
    def _calculate_fallback_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate fallback matrices that should always work."""
        print("🔄 Calculating fallback perspective matrices...")
        
        # Very conservative points that are guaranteed to work
        src = np.float32([
            [0, self.height],           # Bottom-left
            [0, self.height * 0.6],     # Top-left
            [self.width, self.height * 0.6],  # Top-right
            [self.width, self.height]          # Bottom-right
        ])
        
        dst = np.float32([
            [0, self.height],           # Bottom-left
            [0, 0],                     # Top-left
            [self.width, 0],            # Top-right
            [self.width, self.height]   # Bottom-right
        ])
        
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)
        
        print("✅ Fallback matrices calculated")
        return M, Minv
    
    def _test_transformation(self, M: np.ndarray) -> bool:
        """Test if the transformation matrix works correctly."""
        try:
            # Create a test image with known features
            test_img = np.zeros((self.height, self.width), dtype=np.uint8)
            
            # Draw some test patterns
            cv2.rectangle(test_img, 
                         (int(self.width * 0.2), int(self.height * 0.7)),
                         (int(self.width * 0.8), int(self.height * 0.9)),
                         255, -1)
            
            # Apply transformation
            warped = cv2.warpPerspective(test_img, M, (self.width, self.height))
            
            # Check if we got reasonable output
            warped_pixels = np.sum(warped > 0)
            original_pixels = np.sum(test_img > 0)
            
            if warped_pixels == 0 and original_pixels > 0:
                print("❌ Transformation test failed: No output pixels")
                return False
            
            preservation_ratio = warped_pixels / original_pixels if original_pixels > 0 else 0
            print(f"🧪 Transformation test: {preservation_ratio:.1%} preservation")
            
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
    
    def warp_to_birdeye(self, img: np.ndarray) -> np.ndarray:
        """
        Transform image to bird's-eye view with comprehensive error handling.
        
        Args:
            img: Input image to transform
            
        Returns:
            Warped bird's-eye view image
        """
        if not self._validate_input_image(img):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        
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
        warped_pixels = np.sum(warped > 0)
        
        if warped_pixels == 0:
            print("🔄 No pixels in warped image, trying alternative approaches...")
            
            # Try with different border mode
            try:
                warped_alternative = cv2.warpPerspective(
                    original, self.M, (self.width, self.height),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_REPLICATE
                )
                
                if np.sum(warped_alternative > 0) > 0:
                    print("✅ Alternative border mode successful")
                    return warped_alternative
            except:
                pass
        
        # If all else fails, return a minimally processed version
        print("⚠️  Using edge-based fallback")
        edges = cv2.Canny(original, 50, 150) if len(original.shape) == 2 else cv2.Canny(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY), 50, 150)
        return edges
    
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
        except Exception as e:
            print(f"❌ Error in warp_to_camera: {e}")
            return np.zeros(self.img_size, dtype=np.uint8)
    
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
        original_pixels = np.sum(original > 0) if original is not None else 0
        warped_pixels = np.sum(warped > 0) if warped is not None else 0
        
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
            if np.sum(warped > 0) > 0:
                print("✅ Fallback 1 (NEAREST) successful")
                return warped
        except:
            pass
        
        # Strategy 2: Return original image with warning
        print("⚠️  All fallbacks failed, returning original image")
        return img if img is not None else np.zeros((self.height, self.width), dtype=np.uint8)
    
    def get_transformation_info(self) -> dict:
        """Get information about the current transformation setup."""
        det = np.linalg.det(self.M) if self.M is not None else 0
        return {
            "image_size": (self.width, self.height),
            "matrix_determinant": det,
            "matrix_type": "identity" if np.allclose(self.M, np.eye(3)) else "custom"
        }


# Example usage
if __name__ == "__main__":
    # Test the transformer
    transformer = PerspectiveTransformer((640, 480))
    info = transformer.get_transformation_info()
    print("Transformation Info:", info)