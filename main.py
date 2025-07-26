# main.py

import cv2
import torch
import pyttsx3
import argparse
from PIL import Image
import numpy as np
import os

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Speak function
def speak(text):
    print(f"🔊 Speaking: {text}")
    engine.say(text)
    engine.runAndWait()

# Function 1: Live Camera Detection
def live_camera_detection():
    cap = cv2.VideoCapture(0)
    last_spoken = None

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("🎥 Live detection started. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (640, 480))
        results = model(small_frame)

        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]
        spoken_now = set()

        for i in range(len(labels)):
            x1, y1, x2, y2, conf = cords[i][:5]
            if conf > 0.5:
                cls = int(labels[i])
                label = model.names[cls]

                if label not in spoken_now and label != last_spoken:
                    speak(label)
                    last_spoken = label
                    spoken_now.add(label)

        results.render()
        cv2.imshow("Smart Helmet View", results.ims[0])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function 2: Image File Detection
def image_detection(image_path):
    if not os.path.exists(image_path):
        print("❌ Image path not found")
        return
    
    img = Image.open(image_path)
    results = model(img)
    results.print()

    labels = results.names
    detections = results.pandas().xyxy[0]['name'].unique()

    if len(detections) == 0:
        speak("No object detected.")
    else:
        for label in detections:
            speak(label)

    # Show image with detection boxes
    results.render()
    img_with_boxes = np.squeeze(results.ims[0])
    cv2.imshow("Detected Image", img_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Vision Helmet Main Controller")
    parser.add_argument('--mode', type=str, default='live', choices=['live', 'image'],
                        help="Choose mode: 'live' for camera, 'image' for picture detection")
    parser.add_argument('--image', type=str, default='', help="Path to image if using image mode")

    args = parser.parse_args()

    if args.mode == 'live':
        live_camera_detection()
    elif args.mode == 'image':
        image_detection(args.image)
