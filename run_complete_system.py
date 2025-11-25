"""
Main script to run the complete LKA system with TuSimple dataset
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_and_setup_dataset():
    """Check if dataset exists, setup if needed"""
    data_dir = Path("tusimple_data")
    videos_dir = data_dir / "videos"
    
    if videos_dir.exists() and any(videos_dir.glob("*.mp4")):
        print("Dataset already exists.")
        return True
    else:
        print("Dataset not found. Setting up...")
        from tusimple_downloader import TuSimpleDatasetLoader
        loader = TuSimpleDatasetLoader()
        return loader.setup_dataset()

def main():
    print("=== TuSimple LKA System ===")
    print("1. Check/setup dataset")
    print("2. Run LKA on test videos")
    print("3. Generate performance reports")
    
    # Check and setup dataset
    if not check_and_setup_dataset():
        print("Dataset setup failed. Using sample data...")
    
    try:
        from lka_tusimple import TusimpleLKASystem
        lka_system = TusimpleLKASystem()
        lka_system.run_on_all_test_videos()
        
    except Exception as e:
        print(f"Error in main LKA system: {e}")
        print("Falling back to basic test...")
        
        from quick_test import test_lka_on_sample
        test_lka_on_sample()

if __name__ == "__main__":
    main()