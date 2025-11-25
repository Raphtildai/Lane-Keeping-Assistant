from src.main import LKASystem
import os

# Create sample data directory
os.makedirs('data', exist_ok=True)

# For testing, you can use any driving video
# Download a sample video or use your own
video_path = 'data/driving_video.mp4'  # Replace with your video path

if os.path.exists(video_path):
    lka = LKASystem(video_path, 'outputs')
    lka.run()
else:
    print(f"Please place a video file at {video_path}")
    print("You can download sample videos from:")
    print("- TuSimple dataset: https://github.com/TuSimple/tusimple-benchmark")
    print("- Or record your own driving video")