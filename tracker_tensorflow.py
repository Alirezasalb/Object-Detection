# object_tracking_counter.py

import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.applications import EfficientNetB7
from keras.applications.efficientnet import preprocess_input, decode_predictions
#from keras.preprocessing import image      ### deprecated
import time


# COCO class names (SSD MobileNet uses 90 classes, 1-indexed)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


# ----------------------------
# Step 1: Load Models
# ----------------------------

# Load SSD MobileNet V2 from TensorFlow Hub (object detection)
print("Loading object detection model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# Load EfficientNetB7 for classification (optional refinement)
print("Loading EfficientNetB7 for classification...")
classifier = EfficientNetB7(weights='imagenet', include_top=True)

# ----------------------------
# Step 2: Helper Functions
# ----------------------------

def detect_objects(frame):
    """Run object detection on a frame using SSD MobileNet."""
    input_tensor = tf.convert_to_tensor(frame)
    input_tensor = input_tensor[tf.newaxis, ...]
    
    detections = detector(input_tensor)
    
    # Extract detection results
    boxes = detections['detection_boxes'][0].numpy()
    classes = detections['detection_classes'][0].numpy().astype(int)
    scores = detections['detection_scores'][0].numpy()
    
    return boxes, classes, scores

def classify_crop(crop_img):
    """Classify a cropped image using EfficientNetB7."""
    if crop_img.size == 0:
        return "unknown", 0.0
    
    # Resize to 600x600 (EfficientNetB7 input size)
    crop_resized = cv2.resize(crop_img, (600, 600))
    crop_rgb = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2RGB)
    
    # Convert to array using tf.keras.utils
    x = tf.keras.utils.img_to_array(crop_rgb)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)  # This is still valid from keras.applications.efficientnet
    
    preds = classifier.predict(x, verbose=0)
    decoded = decode_predictions(preds, top=1)[0][0]
    label = decoded[1]
    confidence = decoded[2]
    return label, confidence

def convert_box_to_pixels(box, height, width):
    """Convert normalized box [y1, x1, y2, x2] to pixel coordinates."""
    y1, x1, y2, x2 = box
    return int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)

# ----------------------------
# Step 3: Main Tracking & Counting Logic
# ----------------------------

class ObjectTracker:
    def __init__(self, detection_threshold=0.5, classify_with_efficientnet=True):
        self.detection_threshold = detection_threshold
        self.classify_with_efficientnet = classify_with_efficientnet
        self.trackers = []  # List of (tracker, label, id)
        self.next_id = 1
        self.counted_ids = set()
        self.total_count = 0

    def reset_trackers(self):
        self.trackers = []

    def process_frame(self, frame):
        height, width, _ = frame.shape
        original_frame = frame.copy()

        # Re-detect every 10 frames or if no trackers
        if len(self.trackers) == 0 or self.frame_count % 10 == 0:
            self.reset_trackers()
            boxes, classes, scores = detect_objects(frame)

            for i in range(len(scores)):
                if scores[i] >= self.detection_threshold:
                    x1, y1, x2, y2 = convert_box_to_pixels(boxes[i], height, width)
                    # Filter out very small boxes
                    if (x2 - x1) * (y2 - y1) < 500:
                        continue

                    # Create tracker
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))

                    # Get label
                    class_id = int(classes[i])
                    if 1 <= class_id < len(COCO_CLASSES):
                        label = COCO_CLASSES[class_id]
                    else:
                        label = "unknown"
                        
                    if self.classify_with_efficientnet:
                        crop = frame[y1:y2, x1:x2]
                        label, conf = classify_crop(crop)
                        label = f"{label} ({conf:.2f})"

                    self.trackers.append((tracker, label, self.next_id))
                    if self.next_id not in self.counted_ids:
                        self.counted_ids.add(self.next_id)
                        self.total_count += 1
                    self.next_id += 1

        # Update trackers
        active_trackers = []
        for tracker, label, obj_id in self.trackers:
            success, bbox = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ID:{obj_id}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                active_trackers.append((tracker, label, obj_id))
            # Else: tracker lost, drop it

        self.trackers = active_trackers
        self.frame_count += 1

        # Display count
        cv2.putText(frame, f"Total Objects: {self.total_count}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return frame

    def start_counting(self, video_path=0):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        self.frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = self.process_frame(frame)
            cv2.imshow('Object Tracking & Counting', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Final object count: {self.total_count}")

if __name__ == "__main__":
    # Set to True to use EfficientNetB7 for classification (slower)
    USE_EFFICIENTNET = True  # Set to True if you want detailed classification

    tracker = ObjectTracker(
        detection_threshold=0.5,
        classify_with_efficientnet=USE_EFFICIENTNET
    )

    # Use webcam: video_path=0
    # Use video file: video_path="path/to/video.mp4"
    tracker.start_counting(video_path=0)