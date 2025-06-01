🗑️ GarbageClassifier

A real-time object detection system for classifying and sorting garbage using YOLOv8 and a Jetson-based robotic platform. This project uses computer vision to detect categories like plastic, metal, and paper, and then triggers appropriate actions via servos or consumer logic.

---

📸 Demo

Demo showing live detection and servo actuation (coming soon).

---

🧠 Features

🔍 YOLOv8 object detection (via ultralytics library)
🧠 Real-time garbage classification: plastic, metal, paper, and other
⚙️ Servo/actuator control via TrashCom for sorting
🎦 Multi-threaded camera capture (~30 FPS)
⏱️ Detection stability logic to avoid false triggers
💤 Cooldown after each action to prevent repetitive actuation

---

🛠️ Requirements
```
Python 3.8+
Ultralytics YOLOv8
OpenCV
A compatible YOLO model (last.pt)
Jetson device (or mock if testing)
Custom modules:
trash_servo.py (provides TrashCom, CMDS)
jetsonConsumer.py (provides JetsonConsumer)
```
Install dependencies:

`pip install opencv-python ultralytics`
or 
`pip install -r requirements.txt`

---

📂 File Structure

-  garbage_classifier.py      # Main script
-  trash_servo.py             # Hardware interface for trash sorting
-  jetsonConsumer.py          # Jetson GPIO/actuator logic
-  last.pt                    # Trained YOLO model
-  README.md

---
  
🚀 Getting Started

1. Clone the repo
```
git clone https://github.com/yourusername/GarbageClassifier.git

cd GarbageClassifier
```

3. Prepare the YOLO Model
Train or download a YOLOv8 model and name it last.pt. Place it in the root directory.

Ensure your model uses class labels like:

bottle, cup, plastic bag → plastic
can, metal → metal
paper, box, book → paper
You can modify these mappings in classify_label().

3. Run the script
```
python garbage_classifier.py
Press q to quit the video stream.
```
---

⚙️ Customization

Adjust confidence threshold or object size filters:
```
self.CONFIDENCE_THRESHOLD = 0.4
self.MIN_DIAGONAL_PX = 50
self.MAX_DIAGONAL_PX = 700
Change cooldown:
self.POST_ACTION_SLEEP_DURATION = 2.0
```
Add or refine label mappings in `classify_label()`.

---

📦 Future Plans

Add support for image logging
Web UI or dashboard for monitoring
Integration with cloud for analytics
Expand label set for more waste categories

---

🤖 Hardware Integration

TrashCom: sends serial commands to control sorting bins.
JetsonConsumer: contains category-specific action methods like plastic(), metal(), and other() for GPIO or motor control.

---

📄 License

MIT License. See LICENSE for details.
