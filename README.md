# Lane Keep Assist (LKA) System – Classical CV Baseline

> **A real-time Lane Keep Assist (LKA) system using classical computer vision**  
> Implements full pipeline: preprocessing → perspective warp → sliding window → polynomial fitting → temporal smoothing → overlay + HUD + metrics

---

## Project Structure
```text
project/
├── data/                   # Input driving videos
├── calib/                  # (Optional) Camera calibration files
├── src/
│   ├── main.py             # Main pipeline orchestrator
│   ├── preprocess.py       # Color + edge thresholding
│   ├── warp.py             # PerspectiveTransformer (forward → bird’s-eye)
│   ├── lane_fit.py         # LaneDetector: histogram, sliding window, polyfit
│   ├── temporal.py         # TemporalFilter: exponential smoothing
│   ├── overlay.py          # OverlayRenderer: HUD, lane drawing
│   ├── metrics.py          # MetricsCalculator: lateral offset, CSV
│   └── utils.py            # ROI masking (trapezoidal)
├── outputs/
│   ├── annotated_output.mp4
│   └── per_frame_metrics.csv
└── README.md
```
---

## Features

| Feature | Implemented |
|-------|-------------|
| **Left/Right Lane Detection** | Yes |
| **Polynomial Curve Fitting (2nd order)** | Yes |
| **Sliding Window Search** | Yes |
| **Confidence Scoring** | Yes (pixel count + fit residual + temporal consistency) |
| **Temporal Smoothing** | Yes (exponential, confidence-weighted) |
| **Dynamic Perspective Adjustment** | Yes |
| **HUD Overlay** | Yes (colored, dynamic YES/NO) |
| **Lateral Offset (meters)** | Yes |
| **Per-frame CSV Export** | Yes |
| **Debug Visualization** | Yes (`--debug` mode) |

---

## Pipeline Overview

```text
[Input Frame]
     ↓
[Preprocess: HLS + Sobel] → binary mask
     ↓
[ROI Masking (Trapezoid)]
     ↓
[Perspective Warp → Bird’s-eye]
     ↓
[Histogram → Lane Base Peaks]
     ↓
[Sliding Window Search]
     ↓
[Polynomial Fit x = f(y)]
     ↓
[Confidence Calculation]
     ↓
[Temporal Smoothing]
     ↓
[Project Back to Camera View]
     ↓
[Draw: Green/Blue Lines + Polygon + HUD]
     ↓
[Write Frame + CSV]
```
## How to run
## 1. Install Dependencies
```bash
pip install -r requirements.txt
```
## 2. Run the entire system
```bash
python main.py --video data/sample_test_video.mp4 --output results --debug
```
### 3. Run the quick test
```bash
python quick_test.py
```
---
## Output
```text
outputs/annotated_output.mp4 → annotated video
outputs/per_frame_metrics.csv → frame-level metrics
outputs/debug_*.jpg → sliding window debug (if --debug)
```
---
## CSV Format
```text
frame_id,left_detected,right_detected,left_conf,right_conf,lat_offset_m
0,1,1,0.92,0.88,-0.05
1,1,1,0.89,0.91,-0.07
...
```
---
### HUD Display
```text
Road Lane Assist by: Kipchirchir Raphael, LGL7CS
Left:  YES | Conf: 0.92
Right: YES | Conf: 0.88
Lat Offset: -0.05m
```
---
```text
Green = Detected (conf > 0.6)
Orange = Not detected
Yellow = Author credit
Cyan = Lateral offset
```
---
## Implementation Details
```text
Component,Method
Color Thresholding,HLS (S-channel) + LAB (L-channel) + HSV
Edge Detection,Sobel X on grayscale
ROI,Trapezoidal mask (no cropping)
IPM,4-point homography (dynamic adjustment)
Lane Search,Histogram + 9 sliding windows
Curve Fit,"np.polyfit(y, x, 2)"
Confidence,0.5*pixel_count + 0.5*residual
Smoothing,Confidence-weighted exponential filter
Overlay,"cv2.polylines, cv2.fillPoly, cv2.putText"
```
---
## Tested On
- Daytime highway (clear markings)
- Urban roads (faded paint)
- Curved roads
- Shadows (partial success)

## Limitations
- Struggles in heavy rain, low light, or extreme glare
- Faded/dashed lanes reduce confidence
- Shifting camera
- Multi Lane with vertial camera
- Sharp curves may require 3rd-order polynomials
- No camera calibration (assumes ideal lens)

## Future Improvements
- 3rd-order polynomial fitting
- Curvature radius estimation
- Kalman filter for lane tracking
- Deep learning fallback (LaneNet)
- ONNX export for embedded deployment