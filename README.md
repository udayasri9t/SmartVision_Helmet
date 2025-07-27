# 🧠 Smart Vision Helmet for the Blind

Smart Vision Helmet is an AI-based assistive device that helps visually impaired people detect obstacles and navigate safely using real-time object detection and voice guidance. It works without internet and can be mounted on a helmet using Raspberry Pi or ESP32-CAM.

## 🚀 Features

- Real-time camera input (helmet-mounted)
- Object detection using TensorFlow Lite + OpenCV
- Voice alerts using text-to-speech
- Detects people, vehicles, and obstacles
- CSV log export and replay (optional)
- Fully offline, hardware-compatible system

## 🖼️ Preview

### Street View Detection
![Street Detection](./smart_visionhelmet/street_view.png)

### Helmet Camera View (Indoor Test)
![Helmet View](./smart_visionhelmet/helmet_view.png)

## 🛠️ Built With

- Python  
- TensorFlow Lite  
- OpenCV  
- pyttsx3  
- Raspberry Pi / ESP32-CAM (optional hardware)

## ▶️ How to Run

```bash
git clone https://github.com/yourusername/smart-vision-helmet.git
cd smart-vision-helmet
pip install -r requirements.txt
python main.py
