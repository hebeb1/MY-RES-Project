# MY-RES-Project
This is a Magic Mirror prototype as part of the Mid-Year Research Program 2024 at QUT

The Magic Mirror is designed to identify objects in real-time and provide the user with the respective AUSLAN sign, with the aim to help users to learn AUSLAN. The application opens with a ‘Welcome’ page, where the user can see themselves and then they can move into either ‘Learning’ or ‘Game’ mode by pressing on the corresponding button. In ‘Learning’ mode, the user can hold up an object to the camera, it will be identified, and then a video of the AUSLAN sign for the object will overlay on the live feed. Each object that is learnt will be listed on the left-hand side of the screen. In ‘Game’ mode, the AUSLAN sign will again be played over the live feed, and then the user can hold up the correct object to win points. The user can toggle ‘Hints on’ or ‘Hints off’ depending on whether they want the word of the object to be displayed or not, and there is also a ‘Skip’ button that allows the user to skip that object. The signs used in this prototype are in the ‘videos_anz’ folder.

My work is based off of an initial prototype created by another student, Tom Desrumeaux. His prototype has been included in this repository to keep a record of the progression of the application.

UNDERSTANDING THE COMPONENTS
IMPORTS:
-	‘tkinter’: Provides the GUI framework
-	‘PIL’ (Pillow): Used for image processing (converting OpenCV images to a format that ‘tkinter’ can display)
-	‘cv2’ (OpenCV): Handles video capture and image processing
-	‘numpy’: Used for numerical operations (e.g. choosing random objects)
-	‘threading’: Manage threads, allowing video playback in parallel with the main program
-	‘YOLO’ (Ultralytics): The pretrained model for real-time object detection. This program uses the ‘yolov8n.pt’ model
MEMORYGAME CLASS:
-	The ‘MemoryGame’ class initialises the YOLO model for object detection and sets the confidence threshold, this being the parameter that determines the minimum confidence level required for the model to consider a detection valid. This threshold helps filter out low-confidence detections, reducing false positives and ensuring that only more certain detections are used. The initialisation also opens a video capture using the webcam on the device (laptop/PC) and sets up the GUI window with control buttons
GUI COMPONENTS:
-	‘self.root’: Main application window
-	‘self.video_frame’: A ‘tkinter’ frame used to display the video feed
-	‘self.control_frame’: Contains control buttons for interacting with the game modes
MODES AND ACTIONS:
-	Modes:
  o	Welcome: Default mode when the program starts, waiting for the user to select a mode
  o	Learning Mode: The program detects and learns objects from the webcam feed
  o	Game Mode: The user must identify previously learned objects 
-	Buttons:
  o	Learn Button: Switches to learning mode
  o	Game Button: Switches to game mode if there are learned objects
  o	Quit Button: Exits the application
  o	Skip Button: Skips the current object in game mode and selects a new one
  o	Show Text Button: Toggles hints on and off during game mode
OBJECT LEARNING AND GAME LOGIC:
-	Object Detection (‘predict’ method): Uses YOLO to detect objects in the video feed
-	Learning Objects: Detected objects (excludes ‘person’) are added to ‘learned_objects’
-	Video Playback (‘play_video’ method): Plays a video associated with a learned object if it exists
-	Game Mode: The program randomly selects a learned object for the user to identify, and if the object is correct the score increases
HELPER METHODS:
-	‘resize_window_proportionately’: Resizes the window based on the aspect ratio
-	‘toggle_show_text’ and ‘update_hint_button_text’: Manage hint visibility
-	‘choose_next_object’: Selects the next object to be identified in game mode
-	‘skip_object’: Allows the user to skip the current object
