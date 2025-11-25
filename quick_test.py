#!/usr/bin/env python3
"""
Quick test script to demonstrate the LKA system
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

def create_improved_sample_video():
    """Create a better sample video for testing"""
    print("Creating improved sample video for testing...")
    
    # Create a simple synthetic lane video
    output_path = Path("sample_test_video.mp4")
    width, height = 1280, 720
    fps = 20
    duration = 5  # seconds
    total_frames = fps * duration
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    for frame_idx in range(total_frames):
        # Create a darker road background (more realistic)
        frame = np.ones((height, width, 3), dtype=np.uint8) * 80  # Dark gray road
        
        # Add bright white lane markings
        center_x = width // 2
        curve_offset = int(50 * np.sin(frame_idx * 0.1))  # Gentle curve
        
        # Left lane marking (bright white)
        left_lane_x = center_x - 200 + curve_offset
        cv2.line(frame, 
                (left_lane_x, 0), 
                (left_lane_x, height), 
                (255, 255, 255), 12)  # Bright white, thick line
        
        # Right lane marking (bright white)
        right_lane_x = center_x + 200 + curve_offset
        cv2.line(frame, 
                (right_lane_x, 0), 
                (right_lane_x, height), 
                (255, 255, 255), 12)  # Bright white, thick line
        
        # Add subtle road texture
        texture = np.random.normal(0, 5, (height, width, 3)).astype(np.uint8)
        frame = cv2.add(frame, texture)
        
        out.write(frame)
    
    out.release()
    print(f"Improved sample video created: {output_path}")
    return output_path

def test_lka_on_sample():
    """Test the LKA system on a sample video"""
    # Create improved sample video
    # video_path = Path("data/sample_test_video.mp4")
    video_path = Path("data/road_5.mp4")
    if video_path.exists():
        # Use existing video
        print("Using existing sample video...")
    else:
        video_path = create_improved_sample_video()
    
    # Run LKA system with debug mode
    print("Running LKA system on sample video...")
    
    try:
        lka_system = LKASystem(str(video_path), "sample_output", debug=True)
        lka_system.run()
        print("\nTest completed! Check 'sample_output' directory for results.")
    except Exception as e:
        print(f"Error running LKA system: {e}")
        print("Trying without debug mode...")
        
        # Try without debug mode
        try:
            lka_system = LKASystem(str(video_path), "sample_output", debug=False)
            lka_system.run()
            print("\nTest completed without debug mode!")
        except Exception as e2:
            print(f"Error even without debug mode: {e2}")
            print("There may be a fundamental issue with the pipeline.")

if __name__ == "__main__":
    test_lka_on_sample()