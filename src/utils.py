import cv2
import numpy as np

# Provides common utility functions used across multiple modules
# Responsibilities:
#     ROI Management: Handles region of interest extraction
#     Coordinate Conversion: Helper functions for coordinate systems
#     Geometry Utilities: Mathematical helpers for line extensions, etc.
#     Common Operations: Reusable functions used by multiple components
# Key Methods:
#     get_roi_mask(): Creates region of interest mask
#     apply_roi(): Applies ROI to an imag
#     extend_line_to_bottom(): Extends polynomial fits to image boundaries

def get_roi_mask(shape, roi_ratio=0.5):
    """Create region of interest mask (lower half of image)"""
    mask = np.zeros(shape[:2], dtype=np.uint8)
    height, width = shape[:2]
    roi_height = int(height * roi_ratio)
    mask[height - roi_height:height, :] = 255
    return mask

def apply_roi(image, roi_ratio=0.5):
    """Apply region of interest to image"""
    height, width = image.shape[:2]
    roi_height = int(height * roi_ratio)
    roi = image[height - roi_height:height, :]
    return roi, (0, height - roi_height)

def extend_line_to_bottom(y_values, poly_coeffs, img_height):
    """Extend polynomial fit to bottom of image"""
    extended_y = np.linspace(0, img_height - 1, img_height)
    extended_x = poly_coeffs[0]*extended_y**2 + poly_coeffs[1]*extended_y + poly_coeffs[2]
    return extended_x, extended_y