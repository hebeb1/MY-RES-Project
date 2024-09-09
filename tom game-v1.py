'''
THIS IS THE WORK OF TOM DESRUMEAUX
'''
import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
import random

class MemoryGame:
    def __init__(self, capture_index, conf_threshold=0.3):
        self.capture_index = capture_index
        self.model = YOLO("yolov8n.pt")
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.objects_list = ["car", "truck", "bicycle", "motorcycle", "laptop", "cell phone", "tv", "cat", "wine glass", "bottle", "book"]
        self.current_sequence = []
        self.sequence_index = 0
        self.score = 0
        self.max_time = 20  # Initial time for each sequence
        self.time_bonus = 5  # Time added for each correct identification
        self.detection_time_threshold = 1.5
        self.last_detected_time = {}
        self.cap = cv2.VideoCapture(capture_index)
        self.start_time = time.time()  # Initialize the start time
        if not self.cap.isOpened():
            raise Exception("Failed to open video capture.")

    def update_sequence(self):
        new_object = random.choice(self.objects_list)
        self.current_sequence.append(new_object)
        self.sequence_index = 0
        self.start_time = time.time()
        self.last_detected_time = {obj: 0 for obj in self.objects_list}
        print(f"New sequence to show: {self.current_sequence}")

    def predict(self, im0):
        return self.model(im0, conf=self.conf_threshold)

    def verify_object(self, detected_objects):
        if self.current_sequence and detected_objects:
            expected_object = self.current_sequence[self.sequence_index]
            if expected_object in detected_objects:
                current_time = time.time()
                if self.last_detected_time[expected_object] == 0:
                    self.last_detected_time[expected_object] = current_time
                elif current_time - self.last_detected_time[expected_object] >= self.detection_time_threshold:
                    print(f"Correct! Found: {expected_object}")
                    self.start_time += self.time_bonus  # Update the start time to add bonus
                    self.sequence_index += 1
                    if self.sequence_index == len(self.current_sequence):
                        self.score += 1
                        return True
                    self.last_detected_time[expected_object] = 0
        return False

    def display_info(self, im0, results, well_played=False):
        time_left = max(0, int(self.max_time + (self.start_time - time.time())))
        object_text = "Show: " + ", ".join(self.current_sequence[:self.sequence_index+1])
        score_text = f"Score: {self.score}"
        time_text = f"Time left: {time_left} sec"
        cv2.putText(im0, object_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im0, score_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 40, 40), 2, cv2.LINE_AA)
        cv2.putText(im0, time_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (41, 57, 255), 2, cv2.LINE_AA)
        if well_played:
            cv2.putText(im0, "Well Played!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

        # Draw bounding boxes for detected objects
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Get boxes
        class_ids = results[0].boxes.cls.cpu().tolist()  # Class IDs
        names = results[0].names
        for box, cls_id in zip(boxes, class_ids):
            label = names[int(cls_id)]
            if label != "person":
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def run_game(self):
        self.update_sequence()
        while True:
            ret, im0 = self.cap.read()
            if not ret:
                print("Warning: Failed to read frame.")
                continue

            results = self.predict(im0)
            detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().tolist()]

            if self.verify_object(detected_objects):
                if self.sequence_index == len(self.current_sequence):
                    self.display_info(im0, results, well_played=True)
                    cv2.imshow('Memory Game', im0)
                    cv2.waitKey(1500)  # Display "Well Played" message for 1.5 seconds
                    self.update_sequence()

            time_left = self.max_time + (self.start_time - time.time())
            if time_left <= 0:
                print(f"Time's up! Final score: {self.score}")
                break

            self.display_info(im0, results)
            cv2.imshow('Memory Game', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Usage
game = MemoryGame(capture_index=0, conf_threshold=0.6)
game.run_game()
