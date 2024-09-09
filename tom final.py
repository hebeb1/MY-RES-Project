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
        self.last_detection_time = None  # Temps de la dernière détection significative
        self.is_video_playing = False  # Indique si une vidéo est en cours de lecture

    def predict(self, im0):
        results = self.model(im0)
        return results

    def display_fps(self, im0):
        self.end_time = time.time()
        fps = 1 / (self.end_time - self.start_time)
        text = f'FPS: {int(fps)}'
        cv2.putText(im0, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

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
        threshold = 100  # Ajustez ce seuil selon vos observations
        return distance < threshold

    def play_video(self, video_path, image_path):
        self.is_video_playing = True
        video_cap = cv2.VideoCapture(video_path)
        image = cv2.imread(image_path)  # Charger l'image
        if image is not None:
            cv2.imshow('Object Detected Image', image)  # Afficher l'image
        else:
            print(f"Image at {image_path} could not be loaded.")

        for _ in range(2):  # Boucle pour jouer la vidéo 2 fois
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rembobiner la vidéo au début
            while video_cap.isOpened():
                ret, frame = video_cap.read()
                if not ret:
                    break
                cv2.imshow('Object Detected Video', frame)
                if cv2.waitKey(25) & 0xFF == 27:  # Press 'Esc' to exit
                    break

        video_cap.release()
        self.is_video_playing = False
        self.last_detection_time = time.time()  # Mise à jour du temps après la lecture de la vidéo
        cv2.destroyAllWindows()  # Fermer les fenêtres de la vidéo et de l'image ensemble

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index, cv2.CAP_AVFOUNDATION)  # Utilisation du backend CAP_AVFOUNDATION
        if not cap.isOpened():
            raise Exception("Failed to open video capture.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            self.start_time = time.time()
            if not self.is_video_playing:  # Effectuer la détection seulement si aucune vidéo n'est en cours
                ret, im0 = cap.read()
                if not ret:
                    print("Warning: Failed to read frame. Reinitializing camera...")
                    cap.release()
                    cap = cv2.VideoCapture(self.capture_index, cv2.CAP_AVFOUNDATION)  # Réinitialiser la caméra
                    continue
                results = self.predict(im0)
                im0, _ = self.plot_bboxes(results, im0)

                if self.last_detection_time is None or (time.time() - self.last_detection_time >= 3):
                    for i, cls_id in enumerate(results[0].boxes.cls.cpu().tolist()):
                        if results[0].names[cls_id] == "person":
                            person_box = results[0].boxes.xyxy.cpu()[i]
                            for j, cls_id in enumerate(results[0].boxes.cls.cpu().tolist()):
                                if cls_id != i:  # S'assurer que ce n'est pas la même boîte
                                    object_box = results[0].boxes.xyxy.cpu()[j]
                                    obj_name = results[0].names[cls_id]
                                    video_file = f"videos_au/{obj_name}.mp4"  # Assurez-vous que le chemin est correct
                                    image_file = f"images/{obj_name}.png"  # Chemin de l'image dans le dossier "images"

                                    # Jouer la vidéo et afficher l'image si elles existent et l'objet est proche d'une personne
                                    if os.path.exists(video_file) and os.path.exists(image_file) and self.is_object_near_person(object_box, person_box):
                                        self.play_video(video_file, image_file)
                                        break  # Sortir de la boucle après avoir joué la vidéo pour éviter les détections multiples

            self.display_fps(im0)
            cv2.imshow('YOLOv8 Detection', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()


# Here will follow the same code but with each line commented

# import torch
# import numpy as np
# import cv2
# import time
# import os
# from ultralytics import YOLO

# class ObjectDetection:
#     def __init__(self, capture_index):
#         self.capture_index = capture_index  # Index or path to the video capture device or file.
#         self.model = YOLO("yolov8n.pt")  # Loading the YOLO model trained for object detection.
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Device configuration to use GPU if available.
#         self.last_detection_time = None  # Variable to store the timestamp of the last detection.
#         self.is_video_playing = False  # Flag to check if a video is currently playing.

#     def predict(self, im0):
#         results = self.model(im0)  # Applying the YOLO model to detect objects in the image.
#         return results

#     def display_fps(self, im0):
#         self.end_time = time.time()  # Current time at the moment of frame processing.
#         fps = 1 / (self.end_time - self.start_time)  # FPS calculation based on the time interval between frames.
#         text = f'FPS: {int(fps)}'  # Preparing the FPS text for display.
#         cv2.putText(im0, text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)  # Rendering the FPS on the frame.

#     def plot_bboxes(self, results, im0):
#         class_ids = []  # List to store class IDs of detected objects.
#         boxes = results[0].boxes.xyxy.cpu()  # Retrieving bounding box coordinates.
#         clss = results[0].boxes.cls.cpu().tolist()  # Retrieving class IDs of detected objects.
#         names = results[0].names  # Retrieving names of detected classes.
#         for box, cls in zip(boxes, clss):
#             class_ids.append(cls)  # Storing class IDs.
#             label = f"{names[int(cls)]}"  # Creating a label for the object.
#             cv2.rectangle(im0, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)  # Drawing the bounding box.
#             cv2.putText(im0, label, (int(box[0]), int(box[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # Labeling the box.
#         return im0, class_ids  # Returning the image with drawn boxes and the list of class IDs.

#     def is_object_near_person(self, object_box, person_box):
#         # Calculating the center points of both boxes.
#         o_center = ((object_box[0] + object_box[2]) / 2, (object_box[1] + object_box[3]) / 2)
#         p_center = ((person_box[0] + person_box[2]) / 2, (person_box[1] + person_box[3]) / 2)
#         # Calculating Euclidean distance between the center points.
#         distance = np.sqrt((o_center[0] - p_center[0]) ** 2 + (o_center[1] - p_center[1]) ** 2)
#         threshold = 100  # Distance threshold to consider objects as being "near".
#         return distance < threshold  # Return True if objects are near, False otherwise.

#     def play_video(self, video_path, image_path):
#         self.is_video_playing = True  # Set flag to True indicating video playback is in progress.
#         video_cap = cv2.VideoCapture(video_path)  # Start video capture from file.
#         image = cv2.imread(image_path)  # Read an image file.
#         if image is not None:
#             cv2.imshow('Object Detected Image', image)  # Display the image if it is loaded successfully.
#         else:
#             print(f"Image at {image_path} could not be loaded.")  # Error message if the image cannot be loaded.

#         for _ in range(2):  # Loop to play the video twice.
#             video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Rewind video to the start.
#             while video_cap.isOpened():
#                 ret, frame = video_cap.read()  # Read frames from the video.
#                 if not ret:
#                     break  # If no frame is read, break the loop.
#                 cv2.imshow('Object Detected Video', frame)  # Display the frame.
#                 if cv2.waitKey(25) & 0xFF == 27:  # Wait for 'Esc' key press to exit.
#                     break

#         video_cap.release()  # Release the video file or capturing device.
#         self.is_video_playing = False  # Set the flag to False as video playback is finished.
#         self.last_detection_time = time.time()  # Update the last detection time.
#         cv2.destroyAllWindows()  # Close all OpenCV windows.

#     def __call__(self):
#         cap = cv2.VideoCapture(self.capture_index, cv2.CAP_AVFOUNDATION)  # Open video capture on specified device using AVFoundation backend.
#         if not cap.isOpened():  # Check if the video capture device is ready.
#             raise Exception("Failed to open video capture.")  # Raise an exception if device is not opened.
#         cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set video frame width.
#         cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set video frame height.

#         while True:  # Infinite loop to process video frames.
#             self.start_time = time.time()  # Record start time for FPS calculation.
#             if not self.is_video_playing:  # Process frames only if no video is currently being played.
#                 ret, im0 = cap.read()  # Read a frame from the video capture.
#                 if not ret:
#                     print("Warning: Failed to read frame. Reinitializing camera...")  # Warning message if frame is not read.
#                     cap.release()  # Release the capture device.
#                     cap = cv2.VideoCapture(self.capture_index, cv2.CAP_AVFOUNDATION)  # Reinitialize the video capture.
#                     continue
#                 results = self.predict(im0)  # Detect objects in the frame.
#                 im0, _ = self.plot_bboxes(results, im0)  # Draw bounding boxes and get updated image.

#                 # Check if any 'person' object is detected and enough time has passed since last detection.
#                 if self.last_detection_time is None or (time.time() - self.last_detection_time >= 3):
#                     for i, cls_id in enumerate(results[0].boxes.cls.cpu().tolist()):
#                         if results[0].names[cls_id] == "person":
#                             person_box = results[0].boxes.xyxy.cpu()[i]
#                             for j, cls_id in enumerate(results[0].boxes.cls.cpu().tolist()):
#                                 if cls_id != i:  # Ensure not to compare the same object.
#                                     object_box = results[0].boxes.xyxy.cpu()[j]
#                                     obj_name = results[0].names[cls_id]
#                                     video_file = f"videos_au/{obj_name}.mp4"  # Path to the video file.
#                                     image_file = f"images/{obj_name}.png"  # Path to the image file.

#                                     # Check if the video and image files exist and the object is near a person.
#                                     if os.path.exists(video_file) and os.path.exists(image_file) and self.is_object_near_person(object_box, person_box):
#                                         self.play_video(video_file, image_file)  # Play video and show image.
#                                         break  # Exit the loop after playing to avoid multiple detections.

#             self.display_fps(im0)  # Display the current FPS on the frame.
#             cv2.imshow('YOLOv8 Detection', im0)  # Show the processed frame with detections.
#             if cv2.waitKey(1) & 0xFF == ord('q'):  # Wait for 'q' key press to exit.
#                 break
#         cap.release()  # Release the video capture device.
#         cvv.destroyAllWindows()  # Close all OpenCV windows.

# # Creating an instance of the ObjectDetection class and invoking it.
# detector = ObjectDetection(capture_index=0)
# detector()

