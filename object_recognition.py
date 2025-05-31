import cv2
from ultralytics import YOLO
import numpy as np
import torch
from garbage_classifier import load_model

class ObjectRecognizer:
    def __init__(self, model_path=None):
        """
        Initialize the ObjectRecognizer with a YOLO model.
        Args:
            model_path (str, optional): Path to a custom YOLO model. 
                                      If None, uses pretrained 'yolov8n' model.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # if model_path:
        #     self.model = YOLO(model_path)
        # else:
        #     self.model = YOLO('yolov8n')
        self.model = load_model()
        # self.model = YOLO('yolov10s.pt')
   
        
        # Define class labels (modify these according to your model's classes)
        self.class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
        
        # Initialize background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False)

    def get_object_bbox(self, frame):
        """
        Detect moving objects in the frame and return the largest bounding box.
        """
        if frame is None or frame.size == 0:
            return None
            
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Apply some noise reduction
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find the largest contour
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(largest_contour)
                return (x, y, w, h)
        return None

    def detect_objects(self, image_path):
        """
        Detect objects in an image and print their labels.
        Args:
            image_path (str): Path to the image file
        Returns:
            list: List of dictionaries containing detection information
        """
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image at {image_path}")

        # Resize image to 300x300
        image = cv2.resize(image, (300, 300))

        # Run inference
        results = self.model(image)[0]
        detections = []

        # Process results
        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, confidence, class_id = result
            class_name = results.names[int(class_id)]
            
            detection = {
                'label': class_name,
                'confidence': confidence,
                'bbox': [x1, y1, x2, y2]
            }
            detections.append(detection)
            
            # Print detection information
            print(f"Detected {class_name} with {confidence:.2f} confidence")

        return detections

    def detect_from_webcam(self, camera_id=0):
        """
        Perform real-time object detection using webcam feed.
        Args:
            camera_id (int): ID of the camera device (default is 0 for primary webcam)
        """
        # Initialize video capture
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera with ID {camera_id}")

        print("Starting real-time detection... Press 'q' to quit.")

        try:
            while True:
                # Read frame from webcam
                ret, frame = cap.read()
                if not ret or frame is None or frame.size == 0:
                    print("Failed to grab frame")
                    break

                # Create a copy for display (we'll add text to this)
                display_frame = frame.copy()
                
                # Get bounding box of moving object
                bbox = self.get_object_bbox(frame)
                
                try:
                    # Ensure frame is valid before resizing
                    if frame is not None and frame.size > 0:
                        # Resize frame to 300x300 for model input
                        resized_frame = cv2.resize(frame, (300, 300))
                        
                        # Add batch dimension
                        resized_frame = np.expand_dims(resized_frame, axis=0)
                        
                        # Run inference on resized frame
                        predictions = self.model(resized_frame)
                        
                        # Get the predicted class
                        predicted_class_idx = np.argmax(predictions[0])
                        confidence = float(predictions[0][predicted_class_idx])
                        class_name = self.class_labels[predicted_class_idx]
                        
                        # Calculate the center region to highlight (50% of frame dimensions)
                        height, width = frame.shape[:2]
                        center_w = width // 2
                        center_h = height // 2
                        box_w = width // 2  # 50% of width
                        box_h = height // 2  # 50% of height
                        x1 = center_w - box_w // 2
                        y1 = center_h - box_h // 2
                        x2 = center_w + box_w // 2
                        y2 = center_h + box_h // 2
                        
                        # Draw bounding box around the center region
                        cv2.rectangle(display_frame, 
                                    (x1, y1), 
                                    (x2, y2), 
                                    (0, 255, 0), 2)
                        
                        # Draw text above the bounding box
                        text = f"{class_name} ({confidence:.2f})"
                        cv2.putText(display_frame, text,
                                  (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.9, (0, 255, 0), 2)
                        
                        # Print detection
                        print(f"Detected: {class_name} ({confidence:.2f})")
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
                
                # Display the frame with predictions
                cv2.imshow('Object Detection', display_frame)

                # Break the loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            # Clean up
            cap.release()
            cv2.destroyAllWindows()

# Example usage
if __name__ == "__main__":
    # Initialize the recognizer
    recognizer = ObjectRecognizer()
    
    try:
        # Start real-time detection from webcam
        recognizer.detect_from_webcam()
    except Exception as e:
        print(f"Error: {e}")
