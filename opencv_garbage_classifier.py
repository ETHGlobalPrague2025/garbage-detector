import cv2
from ultralytics import YOLO

class GarbageClassifier:
    def __init__(self, model_path='yolov10s.pt'):
        self.model = YOLO(model_path)

        # Mapping COCO labels into your categories
        self.plastic_labels = ['bottle', 'cup', 'plastic']
        self.metal_labels = ['can']  # Assuming can is detected
        self.paper_labels = ['paper', 'box', 'book']
        # "rest" means all other detected objects or undetected garbage

    def classify_label(self, label):
        label_lower = label.lower()
        if any(pl in label_lower for pl in self.plastic_labels):
            return 'plastic'
        elif any(ml in label_lower for ml in self.metal_labels):
            return 'metal'
        elif any(ppl in label_lower for ppl in self.paper_labels):
            return 'paper'
        else:
            return 'rest'

    def process_frame(self, frame):
        results = self.model(frame)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                category = self.classify_label(label)

                conf = float(box.conf[0])
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = xyxy

                # Draw bounding box and label (category + confidence)
                color_map = {
                    'plastic': (0, 255, 0),  # green
                    'metal': (0, 0, 255),    # red
                    'paper': (255, 255, 0),  # cyan-ish
                    'rest': (128, 128, 128)  # gray
                }
                color = color_map.get(category, (255, 255, 255))

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f'{category} {conf:.2f}',
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )
        return frame

    def run_video_stream(self, video_source=0):
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            print("Error: Could not open video source.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            output_frame = self.process_frame(frame)
            cv2.imshow("Garbage Classification", output_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    gc = GarbageClassifier()
    gc.run_video_stream(0)
