'''
THIS IS THE WORK OF HEBE BICKHAM
'''

# PROTOTYPE FOR LEARNING AND GAME MODES (NO AUSLAN)
import torch
import numpy as np
import cv2
from ultralytics import YOLO

class MemoryGame:
    def __init__(self, capture_index, conf_threshold=0.3):
        self.capture_index = capture_index  
        self.model = YOLO("yolov8n.pt")
        self.conf_threshold = conf_threshold
        self.learned_objects = set()  # Set to store learned objects
        self.score = 0
        self.cap = cv2.VideoCapture(capture_index)
        if not self.cap.isOpened():
            raise Exception("Failed to open video capture.")

        # Get the original width and height of the video frames
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create a window named 'Magic Mirror' with normal size
        cv2.namedWindow('Magic Mirror', cv2.WINDOW_NORMAL)
        # Resize the window to the desired size while maintaining the aspect ratio
        self.resize_window_proportionately()

        self.current_object = None
        self.previous_object = None

    def resize_window_proportionately(self, target_width=1280):
        aspect_ratio = self.frame_width / self.frame_height
        new_height = int(target_width / aspect_ratio)
        cv2.resizeWindow('Magic Mirror', target_width, new_height)

    def learn_object(self, im0):
        results = self.predict(im0)
        detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().tolist()]
        
        for obj in detected_objects:
            if obj not in self.learned_objects and obj != "person":  # Exclude "person" class
                self.learned_objects.add(obj)
                print(f"Learned object: {obj}")

        self.display_info(im0, results, mode="Learning Mode")
        cv2.imshow('Magic Mirror', im0)
        
    def predict(self, im0):
        return self.model(im0, conf=self.conf_threshold)

    def verify_object(self, detected_objects):
        for obj in detected_objects:
            if obj == self.current_object and obj in self.learned_objects and obj != "person":  # Exclude "person" class
                return True
        return False

    def display_info(self, im0, results, mode="Game Mode", well_played=False):
        if mode == "Learning Mode":
            mode_text = "Learning Mode: Show objects to learn"
        else:
            mode_text = "Game Mode: Identify learned objects"

        score_text = f"Score: {self.score}"
        
        cv2.putText(im0, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im0, score_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 40, 40), 2, cv2.LINE_AA)
        
        if mode == "Game Mode":
            if self.current_object and self.current_object != "person":  # Exclude "person" class
                cv2.putText(im0, f"Show: {self.current_object}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            else:
                cv2.putText(im0, "Show: (None)", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw learned objects list on the screen during learning mode
        if mode == "Learning Mode":
            y_offset = 150
            cv2.putText(im0, "Learned Objects:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 40
            for obj in self.learned_objects:
                if obj != "person":  # Exclude "person" class
                    cv2.putText(im0, obj, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 40

        # Draw bounding boxes for detected objects (excluding "person" class)
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get boxes
        class_ids = results[0].boxes.cls.cpu().tolist()  # Class IDs
        names = results[0].names
        for box, cls_id in zip(boxes, class_ids):
            label = names[int(cls_id)]
            if label != "person":  # Exclude "person" class
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display "Well Done" if well_played is True
        if well_played:
            cv2.putText(im0, "Well Done!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

    def run(self):
        print("Press 'l' to enter learning mode, 'g' to enter game mode, 'q' to quit.")
        mode = "Learning"  # Default mode

        while True:
            ret, im0 = self.cap.read()
            if not ret:
                print("Warning: Failed to read frame.")
                continue

            # Resize the frame while maintaining aspect ratio
            aspect_ratio = self.frame_width / self.frame_height
            new_w = 1280
            new_h = int(new_w / aspect_ratio)
            im0_resized = cv2.resize(im0, (new_w, new_h))

            if mode == "Learning":
                self.learn_object(im0_resized)
            elif mode == "Game":
                if not self.learned_objects:
                    print("No objects learned yet. Please run learning mode first.")
                    mode = "Learning"
                    continue

                results = self.predict(im0_resized)
                detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().tolist() if results[0].names[int(cls)] != "person"]  # Exclude "person" class
                self.display_info(im0_resized, results)
                cv2.imshow('Magic Mirror', im0_resized)

                # Check if the current object is correctly identified
                if self.current_object and self.current_object != "person":  # Exclude "person" class
                    if self.verify_object(detected_objects):
                        self.score += 1
                        print(f"Correctly identified: {self.current_object}")
                        self.display_info(im0_resized, results, well_played=True)  # Display "Well Done"
                        cv2.imshow('Magic Mirror', im0_resized)
                        cv2.waitKey(500)  # Delay for 0.5 seconds (adjust as needed)
                        self.previous_object = self.current_object
                        self.current_object = None  # Reset current_object after correct identification
                    else:
                        self.display_info(im0_resized, results)  # Update display without delay

                # Pick a new object to ask for if current_object is None
                if self.current_object is None:
                    valid_objects = [obj for obj in self.learned_objects if obj != "person" and obj != self.previous_object]  # Exclude "person" class and previously asked object
                    if valid_objects:
                        self.current_object = np.random.choice(valid_objects)
                    else:
                        self.current_object = np.random.choice([obj for obj in self.learned_objects if obj != "person"])  # Fallback if no valid object found

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('l'):
                mode = "Learning"
                print("Switched to Learning Mode")
            elif key == ord('g'):
                if self.learned_objects:
                    mode = "Game"
                    print("Switched to Game Mode")
                    valid_objects = [obj for obj in self.learned_objects if obj != "person" and obj != self.previous_object]  # Exclude "person" class and previously asked object
                    if valid_objects:
                        self.current_object = np.random.choice(valid_objects)  # Start with a random object
                    else:
                        self.current_object = np.random.choice([obj for obj in self.learned_objects if obj != "person"])  # Fallback if no valid object found
                else:
                    print("No objects learned yet. Please run learning mode first.")

        self.cap.release()
        cv2.destroyAllWindows()

# Usage
game = MemoryGame(capture_index=0, conf_threshold=0.6)
game.run()
