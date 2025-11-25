import kagglehub
import pandas as pd
import os
import cv2
import numpy as np
from pathlib import Path
import json
import zipfile
import shutil

class TuSimpleDatasetLoader:
    def __init__(self, dataset_path="manideep1108/tusimple"):
        self.dataset_path = dataset_path
        self.base_dir = Path("tusimple_data")
        self.base_dir.mkdir(exist_ok=True)
        
    def download_tusimple_data(self):
        """Download TuSimple dataset using correct KaggleHub API"""
        print("Downloading TuSimple dataset...")
        
        try:
            # Download the entire dataset
            download_path = kagglehub.dataset_download(self.dataset_path)
            print(f"Dataset downloaded to: {download_path}")
            
            # List all files in the downloaded directory
            downloaded_files = {}
            for root, dirs, files in os.walk(download_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, download_path)
                    downloaded_files[relative_path] = file_path
                    print(f"Found: {relative_path}")
            
            return download_path, downloaded_files
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            # Fallback: create sample data
            return self._create_sample_data()
    
    def _create_sample_data(self):
        """Create sample data if download fails"""
        print("Creating sample dataset for testing...")
        
        sample_dir = self.base_dir / "sample_data"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a sample video
        self._create_sample_video(sample_dir / "sample_video.mp4")
        
        # Create sample label file
        self._create_sample_labels(sample_dir / "sample_labels.json")
        
        return str(sample_dir), {}
    
    def _create_sample_video(self, video_path):
        """Create a sample lane detection video"""
        width, height = 1280, 720
        fps = 20
        duration = 10  # seconds
        total_frames = fps * duration
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        for frame_idx in range(total_frames):
            # Create road background
            frame = np.ones((height, width, 3), dtype=np.uint8) * 120
            
            # Add dynamic lane markings (simulating curved road)
            center_x = width // 2
            curve_offset = int(100 * np.sin(frame_idx * 0.2))
            
            # Left lane (green in final overlay)
            left_lane_x = center_x - 200 + curve_offset
            cv2.line(frame, 
                    (left_lane_x, 0), (left_lane_x, height), 
                    (200, 200, 200), 15)  # White lane marking
            
            # Right lane (blue in final overlay)
            right_lane_x = center_x + 200 + curve_offset
            cv2.line(frame, 
                    (right_lane_x, 0), (right_lane_x, height), 
                    (200, 200, 200), 15)  # White lane marking
            
            # Add road texture
            texture = np.random.normal(0, 15, (height, width, 3)).astype(np.uint8)
            frame = cv2.add(frame, texture)
            
            # Add some noise to simulate real conditions
            noise = np.random.normal(0, 5, (height, width, 3)).astype(np.uint8)
            frame = cv2.add(frame, noise)
            
            out.write(frame)
        
        out.release()
        print(f"Created sample video: {video_path}")
    
    def _create_sample_labels(self, label_path):
        """Create sample label file"""
        sample_labels = {
            "lanes": [
                [500, 502, 504, 506, 508, 510, 512, 514, 516, 518],  # Left lane
                [700, 702, 704, 706, 708, 710, 712, 714, 716, 718]   # Right lane
            ],
            "h_samples": [240, 250, 260, 270, 280, 290, 300, 310, 320, 330],
            "raw_file": "sample_video.mp4"
        }
        
        with open(label_path, 'w') as f:
            json.dump(sample_labels, f, indent=2)
        
        print(f"Created sample labels: {label_path}")
    
    def extract_and_organize_data(self, download_path, downloaded_files):
        """Extract and organize the dataset"""
        print("Organizing dataset...")
        
        # Create directory structure
        dirs = ['videos', 'labels']
        for dir_name in dirs:
            (self.base_dir / dir_name).mkdir(exist_ok=True)
        
        # Copy or extract files
        if downloaded_files:
            self._organize_downloaded_files(download_path, downloaded_files)
        else:
            self._organize_sample_data()
        
        print("Dataset organization complete!")
    
    def _organize_downloaded_files(self, download_path, downloaded_files):
        """Organize downloaded files"""
        # Look for zip files to extract
        for rel_path, full_path in downloaded_files.items():
            if rel_path.endswith('.zip'):
                dataset_type = 'train' if 'train' in rel_path.lower() else 'test'
                print(f"Extracting {rel_path}...")
                
                with zipfile.ZipFile(full_path, 'r') as zip_ref:
                    zip_ref.extractall(self.base_dir / dataset_type)
            
            elif rel_path.endswith('.json'):
                # Copy label files
                shutil.copy2(full_path, self.base_dir / 'labels')
        
        # Create videos from extracted images
        self._create_videos_from_images()
    
    def _organize_sample_data(self):
        """Organize sample data"""
        sample_dir = self.base_dir / "sample_data"
        
        # Copy sample video to videos directory
        shutil.copy2(sample_dir / "sample_video.mp4", self.base_dir / "videos")
        
        # Copy sample labels to labels directory
        shutil.copy2(sample_dir / "sample_labels.json", self.base_dir / "labels")
    
    def _create_videos_from_images(self):
        """Create videos from image sequences if images exist"""
        for dataset_type in ['train', 'test']:
            dataset_dir = self.base_dir / dataset_type
            if dataset_dir.exists():
                # Look for clip directories
                for clip_dir in dataset_dir.iterdir():
                    if clip_dir.is_dir():
                        self._create_video_from_clip(clip_dir, dataset_type)
    
    def _create_video_from_clip(self, clip_dir, dataset_type):
        """Create video from a clip directory of images"""
        image_files = list(clip_dir.glob('*.jpg'))
        if not image_files:
            return
        
        # Sort images by name
        image_files.sort()
        
        # Read first image to get dimensions
        sample_img = cv2.imread(str(image_files[0]))
        if sample_img is None:
            return
        
        height, width = sample_img.shape[:2]
        
        # Create video
        video_path = self.base_dir / 'videos' / f'{dataset_type}_{clip_dir.name}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, 20.0, (width, height))
        
        for img_path in image_files:
            img = cv2.imread(str(img_path))
            if img is not None:
                out.write(img)
        
        out.release()
        print(f"Created video: {video_path}")
    
    def get_test_videos(self):
        """Get paths to test videos for LKA system"""
        video_dir = self.base_dir / "videos"
        if video_dir.exists():
            test_videos = list(video_dir.glob('*.mp4'))
            return test_videos
        return []
    
    def setup_dataset(self):
        """Complete dataset setup process"""
        print("=== Setting up TuSimple Dataset ===")
        
        # Download data
        download_path, downloaded_files = self.download_tusimple_data()
        
        # Organize data
        self.extract_and_organize_data(download_path, downloaded_files)
        
        # List available test videos
        test_videos = self.get_test_videos()
        print("\n=== Available Test Videos ===")
        for video_path in test_videos:
            print(f"✓ {video_path.name}")
        
        return len(test_videos) > 0

def main():
    # Initialize dataset loader
    loader = TuSimpleDatasetLoader()
    
    # Setup dataset
    success = loader.setup_dataset()
    
    if success:
        print("\nDataset setup completed successfully!")
    else:
        print("\nUsing sample data for testing.")
    
    return loader

if __name__ == "__main__":
    loader = main()