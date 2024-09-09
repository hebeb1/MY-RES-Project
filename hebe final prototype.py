'''
THIS IS THE WORK OF HEBE BICKHAM
'''

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
import threading
from ultralytics import YOLO

class MemoryGame:
    def __init__(self, capture_index, conf_threshold=0.3):
        # Initialise the game
        self.capture_index = capture_index
        self.model = YOLO("yolov8n.pt")     # Load YOLO model
        self.conf_threshold = conf_threshold    # Confidence threshold
        self.learned_objects = set()    # For storing learned objects
        self.score = 0      # Initialise score
        self.cap = cv2.VideoCapture(capture_index)      # Capture video from the specified index
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Reduce the frame rate

        # Check if the video capture was opened successfully
        if not self.cap.isOpened():
            raise Exception("Failed to open video capture.")

        # Get the width and height of the video frames
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Initialise the main application window using Tkinter
        self.root = tk.Tk()
        self.root.title("Magic Mirror")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)     # Handle closing window

        # Create a frame
        self.video_frame = tk.Frame(self.root)
        self.video_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Create a control frame for the buttons
        self.control_frame = tk.Frame(self.video_frame, bg='white')
        self.control_frame.pack(side=tk.TOP, fill=tk.X)

        # Create a label for displaying the video feed
        self.label = tk.Label(self.video_frame)
        self.label.pack(fill=tk.BOTH, expand=True)

        self.mode = "Welcome"   # Initialise mode to Welcome

        # Create buttons for controlling the different modes
        self.learn_button = tk.Button(self.control_frame, text="Learning Mode", command=self.switch_to_learning_mode)
        self.learn_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.game_button = tk.Button(self.control_frame, text="Game Mode", command=self.switch_to_game_mode)
        self.game_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.quit_button = tk.Button(self.control_frame, text="Quit", command=self.quit_game)
        self.quit_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Create button for skipping objects in Game mode
        self.skip_button = tk.Button(self.control_frame, text="Skip", command=self.skip_object, state=tk.DISABLED)
        self.skip_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Button for toggling hints in Game mode
        self.show_text_button = tk.Button(self.control_frame, text="Turn Hints On", command=self.toggle_show_text, state=tk.DISABLED)
        self.show_text_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Variables for managing the current/previous objects
        self.current_object = None
        self.previous_object = None
        self.is_video_playing = False       # Flag to track if a video is currently playing
        self.video_thread = None        # Thread to handle video playback
        self.video_frame_resized = None     # Store resized video frame
        self.show_text_visible = False      # Flag to toggle showing hints
        self.video_cap = None  # Initialize video capture for playing videos

    def resize_window_proportionately(self, target_width=1280):
        # Resize the window proportionately
        aspect_ratio = self.frame_width / self.frame_height
        new_height = int(target_width / aspect_ratio)
        cv2.resizeWindow('Magic Mirror', target_width, new_height)

    def learn_object(self, im0):
        # Use YOLO to detect objects and 'learn' new ones
        results = self.predict(im0)     # Get the results
        detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().tolist()]

        # Learn new objects other than people
        for obj in detected_objects:
            if obj not in self.learned_objects and obj != "person":
                self.learned_objects.add(obj)
                print(f"Learned object: {obj}")
                self.play_video(obj)        # Play associated video

        self.display_info(im0, results, mode="Learning Mode")       # Display learning mode info
        return im0

    def play_video(self, obj_name):
        # Play the video associated with the learned object
        video_file = f"videos_anz/{obj_name}.mp4"
        self.video_cap = cv2.VideoCapture(video_file)

        if not self.video_cap.isOpened():
            # If no video is found, do nothing
            self.is_video_playing = False
            self.video_cap.release()
            return False
        
        # Start the video playback in a separate thread
        self.is_video_playing = True
        if self.video_thread and self.video_thread.is_alive():
            self.is_video_playing = False
            self.video_thread.join()        # Wait for the previous video thread to finish
        self.video_thread = threading.Thread(target=self.play_video_thread)     # Create new video thread
        self.video_thread.start()
        return True

    def play_video_thread(self):
        # Thread to handle video playback
        while self.is_video_playing and self.video_cap.isOpened():
            ret, video_frame = self.video_cap.read()
            if not ret:
                self.is_video_playing = False
                break
            # Resize the video frame to fit the display window
            aspect_ratio = self.frame_width / self.frame_height
            new_w = 1280
            new_h = int(new_w / aspect_ratio)
            self.video_frame_resized = cv2.resize(video_frame, (new_w, new_h))
            cv2.waitKey(30)  # Control the frame rate of the video playback

    def predict(self, im0):
        # Predict objects in the frame using YOLO (object detection)
        return self.model(im0, conf=self.conf_threshold)

    def verify_object(self, detected_objects):
        # Verify if the current object matches one of the detected objects and has been learned
        return any(obj == self.current_object and obj in self.learned_objects and obj != "person" for obj in detected_objects)

    def toggle_show_text(self):
        # Toggle hint visibility
        self.show_text_visible = not self.show_text_visible
        self.update_hint_button_text()

    def update_hint_button_text(self):
        # Update the hint button based on the current state
        self.show_text_button.config(text="Turn Hints Off" if self.show_text_visible else "Turn Hints On")

    def display_info(self, im0, results, mode="Welcome", well_played=False):
        # Display game info on the video
        mode_text = {
            "Learning Mode": "Learning Mode: Show objects to learn",
            "Game Mode": "Game Mode: Identify learned objects",
            "Welcome": "Welcome: Please select a mode"
        }.get(mode, "")

        score_text = f"Score: {self.score}"
        
        cv2.putText(im0, mode_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(im0, score_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 40, 40), 2, cv2.LINE_AA)
        
        # Show hint and text buttons if in Game Mode
        if mode == "Game Mode":
            if self.show_text_visible and self.current_object and self.current_object != "person":
                show_text = f"Show: {self.current_object}"
                cv2.putText(im0, show_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            self.skip_button.config(state=tk.NORMAL)
            self.show_text_button.config(state=tk.NORMAL)

        # Display learned objects if in Learning Mode
        if mode == "Learning Mode":
            y_offset = 120
            cv2.putText(im0, "Learned Objects:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            y_offset += 40
            for obj in self.learned_objects:
                if obj != "person":
                    cv2.putText(im0, obj, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    y_offset += 40

        # Draw bounding boxes around detected objects with names
        if results:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            class_ids = results[0].boxes.cls.cpu().tolist()
            names = results[0].names
            for box, cls_id in zip(boxes, class_ids):
                label = names[int(cls_id)]
                if label != "person":
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(im0, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if well_played:
            # Display message if the user showed the correct object (gains a point)
            cv2.putText(im0, "Well Done!", (200, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4, cv2.LINE_AA)

    def run(self):
        # Start Tkinter main loop
        self.update()       # Update the game state
        self.root.mainloop()    

    def update(self):
        # Read a frame from the video capture
        ret, im0 = self.cap.read()
        if ret:
            im0 = cv2.flip(im0, 1)      # Flip frame for mirror effect
            
            # Resize the frame to fit the window whilst maintaining aspect ratio
            aspect_ratio = self.frame_width / self.frame_height
            new_w = 1280
            new_h = int(new_w / aspect_ratio)
            im0_resized = cv2.resize(im0, (new_w, new_h))

            # Blend the resized frame with the video frame if a video is playing
            if self.is_video_playing and self.video_frame_resized is not None:
                im0_resized = cv2.addWeighted(im0_resized, 0.2, self.video_frame_resized, 0.8, 0)

            # Mode specific processing
            if self.mode == "Learning":
                # Process the frame for learning mode
                im0_resized = self.learn_object(im0_resized)
            elif self.mode == "Game":
                # Check if any objects have been learned
                if not self.learned_objects:
                    print("No objects learned yet. Please run learning mode first.")
                    self.mode = "Learning"
                else:
                    # Run obejct detection on the frame
                    results = self.predict(im0_resized)
                    if results:
                        detected_objects = [results[0].names[int(cls)] for cls in results[0].boxes.cls.cpu().tolist() if results[0].names[int(cls)] != "person"]
                        self.display_info(im0_resized, results, mode="Game Mode")

                        # Check if the detected objects match the current object
                        if self.current_object and self.current_object != "person":
                            if self.verify_object(detected_objects):
                                self.score += 1
                                print(f"Correctly identified: {self.current_object}")
                                self.display_info(im0_resized, results, mode="Game Mode", well_played=True)
                                cv2.waitKey(500)        # Pause for half a second
                                self.previous_object = self.current_object
                                self.current_object = None
                            else:
                                self.display_info(im0_resized, results, mode="Game Mode")

                        # Choose a new object is there is no current object
                        if self.current_object is None:
                            valid_objects = [obj for obj in self.learned_objects if obj != "person" and obj != self.previous_object]
                            if valid_objects:
                                self.current_object = np.random.choice(valid_objects)
                            else:
                                self.current_object = np.random.choice([obj for obj in self.learned_objects if obj != "person"])

                            if self.current_object:
                                self.play_video(self.current_object)
            else:
                # Display welcome screen if no valid mode
                self.display_info(im0_resized, None, mode="Welcome")

            # Convert the frame to ImageTk format and display it
            im0_rgb = cv2.cvtColor(im0_resized, cv2.COLOR_BGR2RGB)
            im0_pil = Image.fromarray(im0_rgb)
            im0_tk = ImageTk.PhotoImage(image=im0_pil)
            self.label.config(image=im0_tk)
            self.label.image = im0_tk

        # Schedule the next update
        self.root.after(50, self.update)

    def switch_to_learning_mode(self):
        # Set to Learning mode and update UI
        self.mode = "Learning"
        print("Switched to Learning Mode")

    def switch_to_game_mode(self):
        # Set to Game mode if objects have been learned
        if self.learned_objects:
            self.mode = "Game"
            print("Switched to Game Mode")
            self.choose_next_object()       # Choose the next object to identify
            self.skip_button.config(state=tk.NORMAL)        # Enable skip button
        else:
            print("No objects learned yet. Please run learning mode first.")

    def choose_next_object(self):
        # Filter valid objects that have a video file to select next object for the game
        valid_objects = [
            obj for obj in self.learned_objects
            if obj != "person" and obj != self.previous_object and self.video_exists(obj)
        ]

        if valid_objects:
            self.current_object = np.random.choice(valid_objects)
            self.play_video(self.current_object)        # Play video for the selected object
        else:
            print("No valid objects with videos found.")
            self.current_object = None

    def video_exists(self, obj_name):
        # Check if a video file for the given object exists
        video_file = f"videos_anz/{obj_name}.mp4"
        return cv2.VideoCapture(video_file).isOpened()

    def skip_object(self):
        # Skip the current object and choose the next one
        if self.current_object:
            self.previous_object = self.current_object
            self.current_object = None
            if self.video_thread and self.video_thread.is_alive():
                self.is_video_playing = False
                self.video_thread.join()        
            self.choose_next_object()

    def play_next_object(self):
        # Choose and play the next object
        valid_objects = [obj for obj in self.learned_objects if obj != "person" and obj != self.previous_object]
        if valid_objects:
            self.current_object = np.random.choice(valid_objects)
        else:
            self.current_object = np.random.choice([obj for obj in self.learned_objects if obj != "person"])

        if self.current_object:
            self.show_text_for_current_object = True
            self.play_video(self.current_object)

    def quit_game(self):
        # Exit the game
        self.root.quit()

    def on_closing(self):
        # Clean up resources when closing the application
        self.cap.release()
        if self.video_cap:
            self.video_cap.release()
        if self.video_thread:
            self.video_thread.join()
        self.root.destroy()

if __name__ == "__main__":
    game = MemoryGame(capture_index=0, conf_threshold=0.6)
    game.run()