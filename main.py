import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
import threading
import time
import os
from PIL import Image, ImageTk


class EyeTrackingVideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Eye Tracking Video Player")
        self.root.geometry("1200x700")

        # Variables
        self.video_path = None
        self.video_player = None
        self.webcam = None
        self.is_playing = False
        self.eye_coords = []
        self.recording = False

        # Eye detection setup
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

        # Create UI
        self.create_ui()

    def create_ui(self):
        # Create frames
        control_frame = tk.Frame(self.root, bg="#f0f0f0", height=50)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        content_frame = tk.Frame(self.root)
        content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        video_frame = tk.Frame(content_frame, bg="black", width=800)
        video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        webcam_frame = tk.Frame(content_frame, bg="black", width=400)
        webcam_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))

        # Control buttons
        self.select_btn = tk.Button(control_frame, text="Seleziona Video", command=self.select_video)
        self.select_btn.pack(side=tk.LEFT, padx=5)

        self.start_btn = tk.Button(control_frame, text="Avvia Video e Tracking", command=self.start_combined,
                                   state=tk.DISABLED)
        self.start_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = tk.Button(control_frame, text="Ferma Tutto", command=self.stop_combined, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.save_btn = tk.Button(control_frame, text="Salva Dati", command=self.save_eye_data, state=tk.DISABLED)
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Labels for video and webcam displays
        self.video_label = tk.Label(video_frame, bg="black")
        self.video_label.pack(fill=tk.BOTH, expand=True)

        self.webcam_label = tk.Label(webcam_frame, bg="black")
        self.webcam_label.pack(fill=tk.BOTH, expand=True)

        # Status label
        self.status_label = tk.Label(self.root, text="Seleziona un video per iniziare", bd=1, relief=tk.SUNKEN,
                                     anchor=tk.W)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Seleziona un video",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )

        if self.video_path:
            self.status_label.config(text=f"Video selezionato: {os.path.basename(self.video_path)}")
            self.start_btn.config(state=tk.NORMAL)

            # Initialize video capture
            self.video_player = cv2.VideoCapture(self.video_path)

            # Read first frame to display
            ret, frame = self.video_player.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (800, 450))
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.video_label.config(image=photo)
                self.video_label.image = photo

    def start_combined(self):
        # Start both video playback and eye tracking
        if not self.is_playing and not self.recording:
            # Start video
            self.is_playing = True

            # Start webcam and eye tracking
            self.recording = True
            self.eye_coords = []  # Reset coordinates
            self.webcam = cv2.VideoCapture(0)  # Open default webcam

            # Update UI
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Video e eye tracking in esecuzione...")

            # Start video playback in a separate thread
            threading.Thread(target=self.play_video, daemon=True).start()

            # Start webcam and eye tracking in a separate thread
            threading.Thread(target=self.track_eyes, daemon=True).start()

    def stop_combined(self):
        # Stop both video playback and eye tracking
        if self.is_playing or self.recording:
            # Stop video
            self.is_playing = False

            # Stop webcam and eye tracking
            self.recording = False
            if self.webcam:
                self.webcam.release()
                self.webcam = None

            # Update UI
            self.start_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.DISABLED)
            self.save_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Video e eye tracking terminati")

    def play_video(self):
        if not self.video_player:
            return

        while self.is_playing:
            ret, frame = self.video_player.read()

            if not ret:
                # Video ended, reset and stop
                self.video_player.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.stop_combined()
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (800, 450))

            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.video_label.config(image=photo)
            self.video_label.image = photo

            # Control playback speed
            time.sleep(1 / 30)  # Adjust for smoother playback

            # Update UI in the main thread
            self.root.update()

    def track_eyes(self):
        if not self.webcam:
            return

        while self.recording:
            ret, frame = self.webcam.read()

            if not ret:
                break

            # Process frame for eye detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # Get current video time
            if self.video_player:
                video_time = self.video_player.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            else:
                video_time = 0

            # Process each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Region of interest for the face
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                # Detect eyes
                eyes = self.eye_cascade.detectMultiScale(roi_gray)

                for (ex, ey, ew, eh) in eyes:
                    # Draw rectangle around the eyes
                    cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

                    # Calculate center of the eye
                    eye_center_x = x + ex + ew // 2
                    eye_center_y = y + ey + eh // 2

                    # Draw center point
                    cv2.circle(frame, (eye_center_x, eye_center_y), 2, (0, 0, 255), 2)

                    # Save eye coordinates with timestamp and video time
                    self.eye_coords.append({
                        'timestamp': time.time(),
                        'video_time': video_time,
                        'eye_x': eye_center_x,
                        'eye_y': eye_center_y
                    })

            # Display the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (400, 300))

            photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.webcam_label.config(image=photo)
            self.webcam_label.image = photo

            # Update UI in the main thread
            self.root.update()

    def save_eye_data(self):
        if not self.eye_coords:
            self.status_label.config(text="Nessun dato da salvare")
            return

        file_path = filedialog.asksaveasfilename(
            title="Salva dati eye tracking",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )

        if file_path:
            with open(file_path, 'w') as f:
                f.write("timestamp,video_time,eye_x,eye_y\n")
                for coord in self.eye_coords:
                    f.write(f"{coord['timestamp']},{coord['video_time']},{coord['eye_x']},{coord['eye_y']}\n")

            self.status_label.config(text=f"Dati salvati in: {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = EyeTrackingVideoPlayer(root)
    root.mainloop()