#!/usr/bin/env python3
"""
Quick test script with the improved video
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add src directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.append(src_dir)

from main import LKASystem

def test_lka_on_sample():
    """Test the LKA system on the better test video"""
    video_path = "better_test_video.mp4"
    
    if not os.path.exists(video_path):
        print("Better test video not found. Creating it...")
        # You can add the create_better_test_video() function here if needed
        return
    
    # Run LKA system with debug mode
    print("Running LKA system on better test video...")
    
    try:
        lka_system = LKASystem(video_path, "sample_output", debug=True)
        lka_system.run()
        print("\nTest completed! Check 'sample_output' directory for results.")
    except Exception as e:
        print(f"Error running LKA system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_lka_on_sample()