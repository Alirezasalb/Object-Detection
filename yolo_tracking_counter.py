import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
from ultralytics import YOLO
import os


class YOLOObjectCounter:
    def __init__(self, model_path='yolov8n.pt', confidence_threshold=0.5):
        print(f"Loading YOLOv8 model: {model_path}...")
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.tracked_objects = set()  # Set of track IDs already counted
        self.track_id_to_class = {}  # Map: track_id -> class_name
        self.total_count = 0
        self.class_names = self.model.names  # {0: 'person', 1: 'bicycle', ...}

    def process_video(self, video_path, output_path=None, show_video=True):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video source: {video_path}")
            return

        if output_path:
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        else:
            out = None

        print("Starting tracking and counting... Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                persist=True,
                verbose=False
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                confidences = results[0].boxes.conf.cpu().numpy()

                for i in range(len(boxes)):
                    x1, y1, x2, y2 = map(int, boxes[i])
                    class_id = class_ids[i]
                    track_id = int(track_ids[i])  # ← Convert to Python int (not np.int64)
                    conf = confidences[i]

                    class_name = self.class_names[class_id]

                    # Record mapping from track_id to class_name
                    if track_id not in self.track_id_to_class:
                        self.track_id_to_class[track_id] = class_name

                    # Count only once per track_id
                    if track_id not in self.tracked_objects:
                        self.tracked_objects.add(track_id)
                        self.total_count += 1

                    # Draw
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{class_name} ID:{track_id}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            cv2.putText(frame, f"Total Objects: {self.total_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            if show_video:
                cv2.imshow("YOLOv8 Tracking & Counting", frame)
            if out:
                out.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

        # 🔥 NEW: Print class-wise breakdown
        print(f"\n✅ Processing complete!")
        print(f"Total unique objects detected: {self.total_count}")

        # Count per class
        class_counts = {}
        for cls in self.track_id_to_class.values():
            class_counts[cls] = class_counts.get(cls, 0) + 1

        print("\nBreakdown by class:")
        for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
            print(f"  {cls}: {count}")

        # Optional: Show full ID → class mapping
        # print("\nFull mapping (ID → Class):")
        # for tid, cls in sorted(self.track_id_to_class.items()):
        #    print(f"  ID {tid}: {cls}")


# ----------------------------
# Run the tracker
# ----------------------------

if __name__ == "__main__":
    # Configuration
    VIDEO_PATH = "D:/projects/trak2/vids/11.mp4"  # Set to 0 for webcam
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