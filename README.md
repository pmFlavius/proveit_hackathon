# 🚗 Vision-First ADAS (Advanced Driver Assistance System)

🏆 **Awarded 4th Place at ProveIT Hackathon by Magna Electronics** 🏆

### 👥 Team: Stoza Code
* **Păvălache Mihai-Flavius**
* **Roșca Alexandru**
* **Denis Șveț**

---

A Level 2 Advanced Driver Assistance System (ADAS) built in 24 hours. It uses a decoupled, Vision-Only architecture to detect lanes, track vehicles, estimate ego-speed, and calculate collision risks in real-time.

## 🏗️ Architecture

The system is built for zero I/O blocking and maximum fault tolerance using a **Producer-Consumer** and **Observer** pattern:

* **The Producer (Perception):** Runs YOLOv8 and OpenCV algorithms on an isolated thread to extract raw spatial data from video frames without blocking the main loop.
* **The Consumer (Logic Server):** Asynchronously calculates physics (Time-To-Collision), determines braking decisions, and logs data to disk.
* **The Observers (Telemetry):** Independent 2D (Pygame) and 3D (Ursina Engine) clients that listen to the Logic Server via UDP ports (5005 & 5006). They use linear interpolation (Lerp) to render a smooth 60 FPS dashboard regardless of the AI's processing speed.

## ✨ Key Features

* **Stable Lane Detection:** The core of our lane tracking relies on applying **Canny Edge Detection** to isolate road markings, followed by **Hough Transforms** to map the geometric lines. We then apply an IIR low-pass filter (Temporal Smoothing) to ignore visual noise like tram tracks.
* **Dynamic Ego-Speed:** Calculates the car's speed entirely through vision by tracking asphalt pixel movement using Optical Flow, removing the need for physical speedometer data.
* **Threat Assessment:** Calculates real-time distance and TTC (Time-To-Collision) for vehicles, pedestrians, and traffic signs.
* **Environment Analysis:** Analyzes HSV spectrums to detect night, fog, and road surface conditions (e.g., wet asphalt).
* **Fail-Safe Design:** If the UI/3D renderers crash, the main AI and logic server continue to run and log decisions uninterrupted.

## 🚀 Quick Start

### Requirements
* Python 3.8+

### Installation
```bash
pip install ultralytics opencv-python numpy pygame ursina
```

*(Note: The YOLOv8 model weights will download automatically on the first run).*

### Running the System
Because the architecture is fully decoupled, the visualization clients and the main AI engine must be run separately so they can communicate simultaneously over UDP.

**1. Start the 2D Dashboard:**
```bash
python renderer2d.py
```

**2. Start the 3D Digital Twin:**
```bash
python renderer3d.py
```

**3. Start the Main AI & Interface:**
```bash
python main_interfata.py
```
*Click "Load Video", select your dashcam footage, and press "Start". The 2D and 3D renderers will instantly come to life once the AI begins broadcasting data.*
