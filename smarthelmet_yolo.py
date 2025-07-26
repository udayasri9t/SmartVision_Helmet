import cv2
import torch
import pyttsx3
import time

# Load YOLOv5s model from ultralytics
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Cannot open camera")
    exit()

print("🎧 Smart Vision Helmet: Live Object Detection + Audio Guidance Started")

last_spoken = None
spoken_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for model input and run detection
    small_frame = cv2.resize(frame, (640, 480))
    results = model(small_frame)

    # Extract labels and bounding box data
    labels = results.xyxyn[0][:, -1]
    cords = results.xyxyn[0][:, :-1]

    for i in range(len(labels)):
        x1, y1, x2, y2, conf = cords[i][:5]
        if conf > 0.5:
            cls = int(labels[i])
            label = model.names[cls]

            # Speak only if object is new and 2 seconds passed
            if label != last_spoken and (time.time() - spoken_time) > 2:
                print(f"🔊 Speaking: {label}")
                engine.say(label)
                engine.runAndWait()
                last_spoken = label
                spoken_time = time.time()

    # Show result on screen
    results.render()
    cv2.imshow("Smart Helmet View", results.ims[0])

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
