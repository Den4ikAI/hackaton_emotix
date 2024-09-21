import sys
import cv2
import numpy as np
import mediapipe as mp
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QProgressBar, QSlider, QSpinBox, QDoubleSpinBox, QFormLayout,
    QFrame, QSizePolicy, QDialog, QRadioButton, QLineEdit, QFileDialog, QComboBox
)
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette
from PyQt6.QtCore import QTimer, Qt, QSize, QTime
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import time
from PIL import Image
import torch
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor

from utils import (
    find_face_and_hands, get_bpm_tells, is_blinking, check_hand_on_face,
    get_avg_gaze, get_lip_ratio, get_face_relative_area, get_area, detect_gaze_change
)

FRAMES_PER_ANALYSIS = 10

# Load the pre-trained model and processor
model_name = "emotion"
model = ViTForImageClassification.from_pretrained(model_name)
processor = ViTImageProcessor.from_pretrained(model_name)
model_path = "stress"
stress_model = ViTForImageClassification.from_pretrained(model_path)
stress_model.eval()

# Define image transformations for the stress model
stress_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_and_crop_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Take the first detected face
        face = gray[y:y+h, x:x+w]  # Use grayscale image
        return face, (x, y, w, h)
    else:
        return None, None

def classify_image(image):
    # Detect and crop face
    face, face_coords = detect_and_crop_face(image)
    
    if face is None:
        return None
    
    # Convert image to monochrome PIL format and add channel
    face_pil = Image.fromarray(face).convert('L')
    face_array = np.array(face_pil)
    face_array = np.expand_dims(face_array, axis=2)  # Add channel
    face_array = np.repeat(face_array, 3, axis=2)  # Repeat channel thrice for RGB
    face_pil = Image.fromarray(face_array)
    
    # Transform image for the model
    inputs = processor(images=face_pil, return_tensors="pt")
    
    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities and class labels
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get top-5 predictions
    top5_prob, top5_catid = torch.topk(probs, 7)
    
    # Format results as a dictionary
    results = {
        model.config.id2label[top5_catid[0][i].item()]: float(top5_prob[0][i].item())
        for i in range(top5_prob.size(1))
    }
    
    return results

class HeartRateChart(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(HeartRateChart, self).__init__(fig)
        self.setParent(parent)
        
        self.hr_times = []
        self.hr_values = []
        self.line, = self.axes.plot(self.hr_times, self.hr_values, color='#2ecc71')
        self.axes.set_ylim(60, 140)
        self.axes.set_xlim(0, 60)
        self.axes.set_facecolor('#f0f0f0')
        self.axes.set_xlabel("Время (сек)", fontsize=10)
        self.axes.set_ylabel("Сердцебиение", fontsize=10)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

    def update_chart(self, new_time, new_value):
        self.hr_times.append(new_time)
        self.hr_values.append(new_value)
        
        while self.hr_times and self.hr_times[0] < new_time - 60:
            self.hr_times.pop(0)
            self.hr_values.pop(0)
        
        self.line.set_data(self.hr_times, self.hr_values)
        self.axes.relim()
        self.axes.autoscale_view()
        self.draw()

class StressChart(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(StressChart, self).__init__(fig)
        self.setParent(parent)
        
        self.stress_times = []
        self.stress_values = []
        self.line, = self.axes.plot(self.stress_times, self.stress_values, color='#e74c3c')
        self.axes.set_ylim(0, 100)
        self.axes.set_xlim(0, 60)
        self.axes.set_facecolor('#f0f0f0')
        self.axes.set_xlabel("Время (сек)", fontsize=10)
        self.axes.set_ylabel("Уровень стресса", fontsize=10)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        fig.tight_layout()

    def update_chart(self, new_time, new_value):
        self.stress_times.append(new_time)
        self.stress_values.append(new_value)
        
        while self.stress_times and self.stress_times[0] < new_time - 60:
            self.stress_times.pop(0)
            self.stress_values.pop(0)
        
        self.line.set_data(self.stress_times, self.stress_values)
        self.axes.relim()
        self.axes.autoscale_view()
        self.draw()

class EmotionBarChart(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(EmotionBarChart, self).__init__(fig)
        self.setParent(parent)
        
        self.axes.set_facecolor('#f0f0f0')
        fig.tight_layout()

    def update_chart(self, emotions):
        self.axes.clear()
        
        # Sort emotions in descending order of values
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        
        # Select top 3 emotions
        top_emotions = sorted_emotions[:3]
        
        # Extract labels and values for top 3 emotions
        labels = [emotion[0] for emotion in top_emotions]
        sizes = [emotion[1] for emotion in top_emotions]
        
        # Set positions for labels on the x-axis
        x_positions = range(len(labels))
        
        # Create a bar chart
        self.axes.bar(x_positions, sizes)
        
        # Set labels on the x-axis
        self.axes.set_xticks(x_positions)
        self.axes.set_xticklabels(labels)
        
        
        # Customize the appearance of the chart
        self.draw()

class StyledProgressBar(QProgressBar):
    def __init__(self, *args, **kwargs):
        super(StyledProgressBar, self).__init__(*args, **kwargs)
        self.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                width: 10px;
                margin: 0.5px;
            }
        """)

class SourceSelectionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор источника видео")
        self.setStyleSheet("""
            QDialog {
                background-color: #2c3e50;
            }
            QLabel {
                color: #ecf0f1;
                font-size: 14px;
            }
            QRadioButton {
                color: #ecf0f1;
                font-size: 14px;
            }
            QComboBox, QLineEdit {
                background-color: #34495e;
                color: #ecf0f1;
                border: 1px solid #2c3e50;
                border-radius: 5px;
                padding: 5px;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: #ecf0f1;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        layout = QVBoxLayout()

        self.camera_radio = QRadioButton("Камера")
        self.file_radio = QRadioButton("Файл")
        self.rtsp_radio = QRadioButton("RTSP поток")

        layout.addWidget(self.camera_radio)
        layout.addWidget(self.file_radio)
        layout.addWidget(self.rtsp_radio)

        self.camera_combo = QComboBox()
        self.populate_camera_list()
        layout.addWidget(self.camera_combo)

        self.file_path = QLineEdit()
        self.file_browse = QPushButton("Выбрать")
        self.file_browse.clicked.connect(self.browse_file)
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.file_path)
        file_layout.addWidget(self.file_browse)
        layout.addLayout(file_layout)

        self.rtsp_url = QLineEdit()
        layout.addWidget(self.rtsp_url)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

        self.camera_radio.toggled.connect(self.update_ui)
        self.file_radio.toggled.connect(self.update_ui)
        self.rtsp_radio.toggled.connect(self.update_ui)

        self.camera_radio.setChecked(True)
        self.update_ui()

    def populate_camera_list(self):
        self.camera_combo.clear()
        for i in range(10):  # Check first 10 camera indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.camera_combo.addItem(f"Camera {i}")
                cap.release()

    def update_ui(self):
        self.camera_combo.setEnabled(self.camera_radio.isChecked())
        self.file_path.setEnabled(self.file_radio.isChecked())
        self.file_browse.setEnabled(self.file_radio.isChecked())
        self.rtsp_url.setEnabled(self.rtsp_radio.isChecked())

    def browse_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov)")
        if file_name:
            self.file_path.setText(file_name)

    def get_source(self):
        if self.camera_radio.isChecked():
            return "camera", self.camera_combo.currentIndex()
        elif self.file_radio.isChecked():
            return "file", self.file_path.text()
        elif self.rtsp_radio.isChecked():
            return "rtsp", self.rtsp_url.text()

class FaceAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ эмоций")
        self.setGeometry(100, 100, 1280, 800)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QLabel {
                color: #2c3e50;
                font-size: 14px;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
            QFrame {
                background-color: white;
                border-radius: 10px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left column
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout, 7)

        # Right column
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout, 3)

        # Video and Emotion Chart frame
        video_emotion_frame = QFrame()
        video_emotion_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        video_emotion_layout = QHBoxLayout(video_emotion_frame)

        # Video display
        video_frame = QFrame()
        video_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        video_layout = QVBoxLayout(video_frame)
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self.video_label.setFixedSize(480, 320)
        video_layout.addWidget(self.video_label)
        video_emotion_layout.addWidget(video_frame)

        # Emotion Pie Chart
        emotion_chart_frame = QFrame()
        emotion_chart_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        emotion_chart_layout = QVBoxLayout(emotion_chart_frame)
        self.emotion_chart = EmotionBarChart(self, width=4, height=2)
        emotion_chart_layout.addWidget(self.emotion_chart)
        video_emotion_layout.addWidget(emotion_chart_frame)

        left_layout.addWidget(video_emotion_frame)

        # Charts (Heart rate and Stress)
        charts_frame = QFrame()
        charts_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        charts_layout = QVBoxLayout(charts_frame)
        
        self.chart = HeartRateChart(self, width=8, height=2)
        charts_layout.addWidget(self.chart)
        
        self.stress_chart = StressChart(self, width=8, height=2)
        charts_layout.addWidget(self.stress_chart)
        
        left_layout.addWidget(charts_frame)

        # Controls
        controls_frame = QFrame()
        controls_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        controls_layout = QVBoxLayout(controls_frame)
        self.start_stop_button = QPushButton("СТАРТ")
        self.start_stop_button.clicked.connect(self.toggle_recording)
        controls_layout.addWidget(self.start_stop_button)

        self.source_button = QPushButton("Выбрать источник")
        self.source_button.clicked.connect(self.select_source)
        
        controls_layout.addWidget(self.source_button)
        self.clear_button = QPushButton("Очистить интерфейс")
        self.clear_button.clicked.connect(self.clear_interface)
        self.clear_button.setStyleSheet("""
            QPushButton {
                background-color: #e74c3c;
                color: white;
                border: none;
                padding: 10px;
                border-radius: 5px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #c0392b;
            }
        """)
        controls_layout.addWidget(self.clear_button)

        right_layout.addWidget(controls_frame)

        # Emotions text
        emotions_frame = QFrame()
        emotions_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        emotions_layout = QVBoxLayout(emotions_frame)
        emotions_title = QLabel("Эмоции")
        emotions_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        emotions_layout.addWidget(emotions_title)
        self.emotions_label = QLabel()
        self.emotions_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        emotions_layout.addWidget(self.emotions_label)
        right_layout.addWidget(emotions_frame)

        # Stress Level Bar
        stress_frame = QFrame()
        stress_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        stress_layout = QVBoxLayout(stress_frame)
        stress_label = QLabel("Уровень стресса")
        stress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stress_layout.addWidget(stress_label)
        self.stress_bar = StyledProgressBar()
        self.stress_bar.setRange(0, 100)
        self.stress_bar.setValue(0)
        self.stress_bar.setFixedHeight(30)
        stress_layout.addWidget(self.stress_bar)
        right_layout.addWidget(stress_frame)

        self.timer_label = QLabel("00:00")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 24px; color: #2c3e50;")
        controls_layout.addWidget(self.timer_label)

        # Spacer to push controls to the top
        right_layout.addStretch()

        # Skip frames for video file
        self.frame_skip = 2

        # Initialize other variables and setup
        self.setup_variables()

    def setup_variables(self):
        # Initialize video capture
        self.cap = None
        self.video_source = None
        # Initialize face_mesh and hands
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = mp.solutions.hands.Hands(
            max_num_hands=2,
            min_detection_confidence=0.7
        )
        
        # Timer for updating the GUI
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        
        self.stopwatch_timer = QTimer(self)
        self.stopwatch_timer.timeout.connect(self.update_stopwatch)
        self.stopwatch_seconds = 0
        # Initialize variables
        self.blinks = [False] * FRAMES_PER_ANALYSIS
        self.hand_on_face = [False] * FRAMES_PER_ANALYSIS
        self.face_area_size = 0
        self.emotion_counts = {}
        self.frame_count = 0
        self.start_time = None
        self.is_recording = False

        # Stress indicators
        self.lip_ratio_values = []
        self.bpm_values = []
        self.blink_count = 0
        self.hand_on_face_count = 0
        self.gaze_values = [0] * FRAMES_PER_ANALYSIS
        self.avg_gaze = None
        self.stress_levels = []
        self.avg_stress = None

    def select_source(self):
        dialog = SourceSelectionDialog(self)
        if dialog.exec():
            source_type, source_value = dialog.get_source()
            self.video_source = (source_type, source_value)
            self.setup_video_capture()

    def setup_video_capture(self):
        if self.cap:
            self.cap.release()

        if self.video_source[0] == "camera":
            self.cap = cv2.VideoCapture(self.video_source[1])
        elif self.video_source[0] == "file":
            self.cap = cv2.VideoCapture(self.video_source[1])
        elif self.video_source[0] == "rtsp":
            self.cap = cv2.VideoCapture(self.video_source[1])

        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            print("Error opening video source")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        if not self.cap or not self.cap.isOpened():
            print("Please select a valid video source first.")
            return

        self.is_recording = True
        self.start_time = time.time()
        self.timer.start(30)  # Update every 30 ms
        self.stopwatch_timer.start(1000)  # Update every second
        self.start_stop_button.setText("СТОП")
        self.stopwatch_seconds = 0
        self.timer_label.setText("00:00")
        self.frame_count = 0
        self.chart.hr_times.clear()
        self.chart.hr_values.clear()
        self.stress_chart.stress_times.clear()
        self.stress_chart.stress_values.clear()
        self.stress_bar.setValue(0)
        self.blinks = [False] * FRAMES_PER_ANALYSIS * 4
        self.hand_on_face = [False] * FRAMES_PER_ANALYSIS
        self.face_area_size = 0
        self.emotion_counts = {}
        self.frame_count = 0
        self.lip_ratio_values = []
        self.bpm_values = []
        self.blink_count = 0
        self.hand_on_face_count = 0
        self.avg_gaze = None
        self.stress_levels = []

    def stop_recording(self):
        self.is_recording = False
        self.timer.stop()
        self.stopwatch_timer.stop()
        self.start_stop_button.setText("СТАРТ")
        self.update_average_stress()
        self.update_emotion_pie_chart()

    def clear_interface(self):
        # Stop recording if it's in progress
        if self.is_recording:
            self.stop_recording()

        # Reset timer
        self.stopwatch_seconds = 0
        self.timer_label.setText("00:00")

        # Clear charts
        self.chart.hr_times.clear()
        self.chart.hr_values.clear()
        self.chart.update_chart(0, 0)
        self.stress_chart.stress_times.clear()
        self.stress_chart.stress_values.clear()
        self.stress_chart.update_chart(0, 0)

        # Reset stress bar
        self.stress_bar.setValue(0)
        self.stress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #bdc3c7;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #2ecc71;
                width: 10px;
                margin: 0.5px;
            }
        """)

        # Clear emotion chart
        self.emotion_chart.update_chart({})

        # Reset variables
        self.blinks = [False] * FRAMES_PER_ANALYSIS * 4
        self.hand_on_face = [False] * FRAMES_PER_ANALYSIS
        self.face_area_size = 0
        self.emotion_counts = {}
        self.frame_count = 0
        self.lip_ratio_values = []
        self.bpm_values = []
        self.blink_count = 0
        self.hand_on_face_count = 0
        self.avg_gaze = None
        self.stress_levels = []

        # Clear emotion label
        self.emotions_label.setText("")

        # Reset video label
        self.video_label.clear()

    def update_stopwatch(self):
        self.stopwatch_seconds += 1
        minutes = self.stopwatch_seconds // 60
        seconds = self.stopwatch_seconds % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")


    def update_frame(self):
        if not self.cap or not self.cap.isOpened():
            return

        current_time = time.time() - self.start_time
        if current_time >= 60:  # Stop after 1 minute
            self.stop_recording()
            return
        
        # Skip frames based on frame_skip value
        if self.video_source[0] == "file":
            for _ in range(self.frame_skip - 1):
                self.cap.read()
                self.frame_count += 1
        ret, frame = self.cap.read()
        if not ret:
            self.stop_recording()
        if ret:
            # Resize frame to 480x320
            frame = cv2.resize(frame, (480, 320))
            # Process the frame
            face_landmarks, hands_landmarks = find_face_and_hands(frame, self.face_mesh, self.hands)
            
            if face_landmarks:
                face = face_landmarks.landmark
                self.face_area_size = get_face_relative_area(face)

                # Get gaze direction
                avg_gaze = get_avg_gaze(face)
                self.avg_gaze = avg_gaze

                # Get heart rate data
                cheekL = get_area(frame, False, topL=face[449], topR=face[350], bottomR=face[429], bottomL=face[280])
                cheekR = get_area(frame, False, topL=face[121], topR=face[229], bottomR=face[50], bottomL=face[209])
                bpm_display = get_bpm_tells(cheekL, cheekR, None)

                # Update heart rate chart
                try:
                    bpm_value = float(bpm_display.split(':')[1].split()[0])
                    self.bpm_values.append(bpm_value)
                    self.chart.update_chart(current_time, bpm_value)
                except (IndexError, ValueError):
                    pass

                # Update blinks and hand on face
                is_blinking_now = is_blinking(face)
                if is_blinking_now:
                    self.blink_count += 1
                self.blinks = self.blinks[1:] + [is_blinking_now]
                
                is_hand_on_face = check_hand_on_face(hands_landmarks, face)
                if is_hand_on_face:
                    self.hand_on_face_count += 1
                self.hand_on_face = self.hand_on_face[1:] + [is_hand_on_face]

                # Get lip ratio
                lip_ratio = get_lip_ratio(face)
                self.lip_ratio_values.append(lip_ratio)

                # Get emotion using the custom model
                emotions = classify_image(frame)
                if emotions:
                    emotion_translations = {
                        "sad": "грусть",
                        "disgust": "отвращение",
                        "angry": "злость",
                        "neutral": "нейтральность",
                        "fear": "страх",
                        "surprise": "удивление",
                        "happy": "радость"
                    }
                    
                    emotion_text = "Обнаруженные эмоции в кадре:\n"
                    for emotion, probability in emotions.items():
                        translated_emotion = emotion_translations.get(emotion, emotion)
                        emotion_text += f"{translated_emotion}: {probability:.2%}\n"
                        self.emotion_counts[emotion] = self.emotion_counts.get(emotion, 0) + probability
                    self.emotions_label.setText(emotion_text)

                # Update frame count
                self.frame_count += 1
                
                # Detect stress and update stress chart
                stress_level = self.detect_stress(frame, face)
                stress_level_rule = self.detect_stress_rule()
                stress_level = stress_level * stress_level_rule
                stress_level = stress_level if stress_level <= 1.0 else 1.0
                self.stress_levels.append(stress_level)
                self.stress_chart.update_chart(current_time, stress_level * 100)

            # Convert frame to QPixmap and display
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))
            


    def detect_stress(self, image, face_landmarks):
        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get face bounding box
        h, w, _ = image.shape
        face_points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks]
        left = min(point[0] for point in face_points)
        top = min(point[1] for point in face_points)
        right = max(point[0] for point in face_points)
        bottom = max(point[1] for point in face_points)
        
        # Extract the face
        face_image = Image.fromarray(image[top:bottom, left:right])
        
        # Transform the image
        input_tensor = stress_transform(face_image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            outputs = stress_model(input_tensor)
        
        # Get prediction
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = probabilities[0][0].item()
        
        # Return result
        return confidence

    def detect_stress_rule(self):
        # Calculate individual stress factors
        bpm_stress = 0
        if self.bpm_values:
            avg_bpm = np.mean(self.bpm_values[-10:])
            if avg_bpm > 100:
                bpm_stress = 1.1

        lip_stress = 0
        if self.lip_ratio_values:
            avg_lip_ratio = np.mean(self.lip_ratio_values[-10:])
            if avg_lip_ratio < 0.3:
                lip_stress = 1.01

        blink_stress = 1.05 if self.blink_count > 8 else 0
        hand_stress = 1.05 if self.hand_on_face_count > 2 else 0

        gaze_stress = 0
        if self.gaze_values:
            gaze_change = detect_gaze_change(self.avg_gaze)
            if gaze_change < 0.1:
                gaze_stress = 1.05

        # Calculate total stress for this frame
        frame_stress = bpm_stress + lip_stress + blink_stress + hand_stress + gaze_stress


        return frame_stress
    

    def update_average_stress(self):
        if self.stress_levels:
            avg_stress = np.mean(self.stress_levels) * 100
            self.stress_bar.setValue(int(avg_stress))
            if avg_stress < 40:
                self.stress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid #bdc3c7;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: #2ecc71;
                        width: 10px;
                        margin: 0.5px;
                    }
                """)
            elif avg_stress < 70:
                self.stress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid #bdc3c7;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: #f1c40f;
                        width: 10px;
                        margin: 0.5px;
                    }
                """)
            else:
                self.stress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 2px solid #bdc3c7;
                        border-radius: 5px;
                        text-align: center;
                    }
                    QProgressBar::chunk {
                        background-color: #e74c3c;
                        width: 10px;
                        margin: 0.5px;
                    }
                """)

    def update_emotion_pie_chart(self):
        if self.emotion_counts:
            total = sum(self.emotion_counts.values())
            averaged_emotions = {emotion: count / total for emotion, count in self.emotion_counts.items()}
            
            emotion_translations = {
                "sad": "грусть",
                "disgust": "отвращение",
                "angry": "злость",
                "neutral": "нейтральность",
                "fear": "страх",
                "surprise": "удивление",
                "happy": "радость"
            }
            
            translated_emotions = {emotion_translations[emotion]: count for emotion, count in averaged_emotions.items()}
            
            self.emotion_chart.update_chart(translated_emotions)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceAnalysisApp()
    window.show()
    sys.exit(app.exec())