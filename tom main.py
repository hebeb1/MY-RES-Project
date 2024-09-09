'''
THIS IS THE WORK OF TOM DESRUMEAUX
'''
import torch
import numpy as np
import cv2
import time
import os
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = YOLO("yolov8n.pt")  # Assurez-vous d'avoir le modèle correct
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.last_detection_time = None
        self.is_video_playing = False

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        end_time = time.time()
        fps = 1 / (end_time - self.start_time)
        cv2.putText(im0, f'FPS: {int(fps)}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

    def plot_bboxes(self, results, im0):
        class_ids = []
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            label = f"{names[int(cls)]}"
            cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
            cv2.putText(im0, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        return im0, class_ids

    def is_object_near_person(self, object_box, person_box):
        o_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
        p_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
        distance = np.sqrt((o_center[0] - p_center[0]) ** 2 + (o_center[1] - p_center[1]) ** 2)
        threshold = 100
        return distance < threshold

    def play_video(self, video_cap, im0, pos_x, pos_y):
        ret, frame = video_cap.read()
        if ret:
            frame_resized = cv2.resize(frame, (320, 240))  # Dimension de la vidéo en incrustation
            y_end = min(pos_y + 240, im0.shape[0])
            x_end = min(pos_x + 320, im0.shape[1])
            im0[pos_y:y_end, pos_x:x_end] = frame_resized[:y_end-pos_y, :x_end-pos_x]

    def __call__(self):
        cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)
        cv2.setWindowProperty('YOLOv8 Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cap = cv2.VideoCapture(self.capture_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            raise Exception("Failed to open video capture.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

        while True:
            self.start_time = time.time()
            ret, im0 = cap.read()
            if not ret:
                print("Warning: Failed to read frame. Reinitializing camera...")
                cap.release()
                cap = cv2.VideoCapture(self.capture_index, cv2.CAP_AVFOUNDATION)
                continue

            results = self.predict(im0)
            im0, _ = self.plot_bboxes(results, im0)

            if self.last_detection_time is None or (time.time() - self.last_detection_time >= 3):
                for i, cls_id in enumerate(results[0].boxes.cls.cpu().tolist()):
                    if results[0].names[cls_id] == "person":
                        person_box = results[0].boxes.xyxy.cpu()[i]
                        for j, cls_id in enumerate(results[0].boxes.cls.cpu().tolist()):
                            if cls_id != i:
                                object_box = results[0].boxes.xyxy.cpu()[j]
                                obj_name = results[0].names[cls_id]
                                video_file = f"videos_au/{obj_name}.mp4"
                                image_file = f"images/{obj_name}.png"

                                if os.path.exists(video_file) and os.path.exists(image_file):
                                    video_cap = cv2.VideoCapture(video_file)
                                    image = cv2.imread(image_file)
                                    image_resized = cv2.resize(image, (320, 240))  # Dimension de l'image en incrustation
                                    x_img = min(1600, im0.shape[1] - 320)
                                    y_img = min(840, im0.shape[0] - 240)
                                    im0[y_img:y_img + 240, x_img:x_img + 320] = image_resized
                                    if self.is_object_near_person(object_box, person_box):
                                        self.play_video(video_cap, im0, 1600, 0)  # Position de la vidéo en haut à droite
                                    video_cap.release()
                                    break

            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Utilisation
detector = ObjectDetection(capture_index=0)
detector()









import torch
import numpy as np
import cv2
import time
from ultralytics import YOLO
import random

class MemoryGame:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.model = YOLO("yolov8n.pt")  # Make sure to have the correct model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.objects_list = ["bottle", "cell phone", "book"]
        self.current_sequence = []
        self.sequence_index = 0  # Index to track current object in sequence
        self.score = 0
        self.max_time = 20  # seconds to respond
        self.detection_time_threshold = 1.5  # seconds object must be shown
        self.last_detected_time = {}
        self.cap = cv2.VideoCapture(capture_index)
        if not self.cap.isOpened():
            raise Exception("Failed to open video capture.")

    def update_sequence(self):
        new_object = random.choice(self.objects_list)
        self.current_sequence.append(new_object)
        self.sequence_index = 0  # Reset sequence index for new round
        self.start_time = time.time()  # Restart timer
        self.last_detected_time = {obj: 0 for obj in self.objects_list}
        print(f"New sequence to show: {self.current_sequence}")

    def predict(self, im0):
        results = self.model(im0)
        return results

    def verify_object(self, detected_objects):
        if self.current_sequence and detected_objects:
            expected_object = self.current_sequence[self.sequence_index]
            if expected_object in detected_objects:
                current_time = time.time()
                if self.last_detected_time[expected_object] == 0:
                    self.last_detected_time[expected_object] = current_time
                elif current_time - self.last_detected_time[expected_object] >= self.detection_time_threshold:
                    print(f"Correct! Found: {expected_object}")
                    self.sequence_index += 1  # Move to the next object in sequence
                    if self.sequence_index == len(self.current_sequence):  # If end of sequence, prepare for next round
                        self.score += 1
                        return True
                    self.last_detected_time[expected_object] = 0  # Reset for next object
        return False

    def display_info(self, im0, results, well_played=False):
        object_text = "Show: " + ", ".join(self.current_sequence[:self.sequence_index+1])
        score_text = f"Score: {self.score}"
        time_text = f"Time left: {int(self.max_time - (time.time() - self.start_time))} sec"
        cv2.putText(im0, object_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(im0, score_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(im0, time_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        if well_played:
            cv2.putText(im0, "Well Played!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

    def run_game(self):
        self.update_sequence()  # Start with the first full sequence
        while True:
            ret, im0 = self.cap.read()
            if not ret:
                print("Warning: Failed to read frame.")
                break

            results = self.predict(im0)
            detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().tolist()]

            if self.verify_object(detected_objects):
                if self.sequence_index == len(self.current_sequence):  # Completed current sequence
                    self.display_info(im0, results, well_played=True)
                    cv2.imshow('Memory Game', im0)
                    cv2.waitKey(1500)  # Display "Well Played" message for 1.5 seconds
                    self.update_sequence()  # Add new object and continue
                continue

            if time.time() - self.start_time > self.max_time:
                print(f"Time's up! Final score: {self.score}")
                break

            self.display_info(im0, results)
            cv2.imshow('Memory Game', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

# Usage
game = MemoryGame(capture_index=0)
game.run_game()
