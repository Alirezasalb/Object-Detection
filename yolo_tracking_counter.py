# yolo_tracking_counter.py

import cv2
import numpy as np
from ultralytics import YOLO
import os

class YOLOObjectCounter:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        """
        Initialize YOLOv8 tracker and counter.
        :param model_path: YOLOv8 model (e.g., 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt')
        :param confidence_threshold: Minimum confidence to consider a detection
        """
        print(f"Loading YOLOv8 model: {model_path}...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracked_objects = set()  # Store unique track IDs that have been counted
        self.total_count = 0
        self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}

    def process_video(self, video_path, output_path=None, show_video=True):
        """
        Process video for object tracking and counting.
        :param video_path: Path to input video (or 0 for webcam)
        :param output_path: Optional path to save output video
        :param show_video: Whether to display video in real-time
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video source: {video_path}")
            return

        # Get video properties for saving
        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None

        print("Starting tracking and counting... Press 'q' to quit.")
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 tracking
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                persist=True,  # Keep tracking across frames
                verbose=False
            )

            # Reset count for this frame (we'll accumulate unique IDs)
            current_frame_ids = set()

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    class_id = class_ids[i]
                    track_id = track_ids[i]
                    conf = confidences[i]

                    # Get class name
                    class_name = self.class_names[class_id]

                    # Count this object if not seen before
                    if track_id not in self.tracked_objects:
                        self.tracked_objects.add(track_id)
                        self.total_count += 1

                    current_frame_ids.add(track_id)

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ID:{track_id}"
                    cv2.putText(
                        frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                    )

            # Display total count
            cv2.putText(
                frame,
                f"Total Objects: {self.total_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

            # Show frame
            if show_video:
                cv2.imshow("YOLOv8 Tracking & Counting", frame)

            # Save frame
            if out:
                out.write(frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        # Cleanup
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Processing complete!")
        print(f"Total unique objects detected: {self.total_count}")
        print(f"Unique tracked IDs: {sorted(self.tracked_objects)}")

# ----------------------------
# Run the tracker
# ----------------------------

if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "D:/projects/trak2/sample_video.mp4"  # Set to 0 for webcam
    OUTPUT_PATH = None  # e.g., "output_tracked.mp4" to save
    MODEL = "yolov8n.pt"  # Options: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x

    # Validate video path if not webcam
    if VIDEO_PATH != 0 and not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        exit()

    # Start tracking
    counter = YOLOObjectCounter(model_path=MODEL, confidence_threshold=0.5)
    counter.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        show_video=True
    )