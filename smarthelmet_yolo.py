import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize TTS engine
engine = pyttsx3.init()

# Load YOLOv8 small model
model = YOLO('yolov8n.pt')  # auto-downloads model

# Set classes of interest
TARGET_CLASSES = {'person', 'car', 'truck', 'bus', 'bicycle', 'motorcycle'}

last_alert = None

cap = cv2.VideoCapture(0)  # Change to 1 if using external camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, verbose=False)[0]
    labels = results.names
    boxes = results.boxes.xyxy
    classes = results.boxes.cls.cpu().numpy()

    alert_text = None
    for cls in classes:
        name = labels[int(cls)]
        if name in TARGET_CLASSES:
            alert_text = f"{name} ahead"
            break

    if alert_text and alert_text != last_alert:
        engine.say(alert_text)
        engine.runAndWait()
        last_alert = alert_text

    # Draw bounding boxes
    annotated = results.plot()
    cv2.imshow("Smart Helmet View", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
