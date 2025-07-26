import cv2
import numpy as np
import pyttsx3

# Load YOLOv5 ONNX model
net = cv2.dnn.readNet("yolov5n.onnx")

# Load and check test image
image = cv2.imread("test.jpg")
if image is None:
    print("❌ ERROR: 'test.jpg' not found or unreadable.")
    exit()

# Prepare input blob
blob = cv2.dnn.blobFromImage(image, 1/255.0, (640, 640), swapRB=True, crop=False)
net.setInput(blob)

# Run inference
outputs = net.forward()

# Load text-to-speech engine
engine = pyttsx3.init()

# Detection threshold
CONF_THRESHOLD = 0.3

# COCO class labels
CLASSES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
    "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]

# Parse outputs (NCHW)
for output in outputs:
    for det in output:
        # det: [x, y, w, h, conf, class_scores...]
        scores = det[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > CONF_THRESHOLD:
            label = CLASSES[class_id]
            print(f" Detected: {label} ({confidence:.2f})")

            # Speak out the label
            engine.say(f"{label} ahead")
            engine.runAndWait()

print("✅ Detection done.")
