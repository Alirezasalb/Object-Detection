# YOLOv8 Object Tracking and Counting

#### A real-time object tracking and counting system using **YOLOv8** (Ultralytics) with persistent ID assignment and class-wise statistics.



## ✨ Features

- Real-time object detection and tracking using **YOLOv8 + ByteTrack**
- Counts **each unique object only once** using track IDs
- Displays class names (e.g., `car`, `person`, `truck`) with IDs
- Shows **total count** and **class-wise breakdown** at the end
- Supports **video files** or **webcam**
- Optional: Save annotated output video

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/Alirezasalb/Object-Detection.git
cd yolo-object-tracking-counter
```
### 2. Install dependencies
```bash
pip install -r requirements.txt
```
### 3. Run the tracker

```bash
# On a video file
python yolo_tracking_counter.py --video "path/to/your/video.mp4"

# On webcam
python yolo_tracking_counter.py --video 0

# Save output video
python yolo_tracking_counter.py --video input.mp4 --output output_tracked.mp4
```

## Requirements 

   - Python ≥ 3.8 
   - Windows / Linux / macOS
   - (Optional) CUDA-compatible GPU for faster inference
    


## Output Example

### At the end of processing:
```bash
✅ Processing complete!
Total unique objects detected: 23

Breakdown by class:
  car: 14
  truck: 5
  bus: 3
  person: 1
```

## Customization 

### Edit `yolo_tracking_counter.py` to: 

#### Change model (`yolov8n.pt`, `yolov8s.pt`, etc.)
#### Adjust confidence threshold
#### Filter specific classes
#### Add counting lines (contact author for advanced features)
     

## License 

MIT License 