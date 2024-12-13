# Стандартные библиотеки
import sys
import time
import warnings
from typing import Dict, Tuple, Optional, Union
# Сторонние библиотеки
import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton,
    QFrame, QDialog, QComboBox
)

from torchvision import transforms
from transformers import AutoModelForImageClassification, AutoImageProcessor
from emotixV4 import EmotixV4Emotion
from qt_elements import HeartRateChart, StressChart, EmotionBarChart, StyledProgressBar, SourceSelectionDialog, EmotionsChart, ModelSelectionDialog
# Настройки
warnings.simplefilter("ignore", UserWarning)
from utils import (
    find_face_and_hands, is_blinking, check_hand_on_face,
    get_avg_gaze, get_lip_ratio, get_face_relative_area, calculate_gaze_score
)
from pulse_detector import  PulseDetector
FRAMES_PER_ANALYSIS = 10


class StressModel:
    def __init__(self, model_path) -> None:
        self.stress_model = AutoModelForImageClassification.from_pretrained(model_path)
        self.stress_model.eval()
        self.stress_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_stress(self, image, face_landmarks):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        h, w, _ = image.shape
        face_points = [(int(landmark.x * w), int(landmark.y * h)) for landmark in face_landmarks]
        left = min(point[0] for point in face_points)
        top = min(point[1] for point in face_points)
        right = max(point[0] for point in face_points)
        bottom = max(point[1] for point in face_points)
        
        face_image = Image.fromarray(image[top:bottom, left:right])
        
        input_tensor = self.stress_transform(face_image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.stress_model(input_tensor)
        
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        confidence = probabilities[0][0].item()

        return confidence

class EmotionModel:
    def __init__(self, checkpoint_path) -> None:
        self.emotixv4emotion = EmotixV4Emotion(checkpoint_path = checkpoint_path)

    def detect_face(self, image: np.ndarray, coords) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        faces = [coords]
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            return face, (x, y, w, h)
        return None, None

    def predict(self, image: np.ndarray, top_k: int = 5, coords: tuple = (None,None,None,None)) -> Optional[Dict[str, float]]:
        if image is None:
            return None
        # Detect and crop face
        face_img, _ = self.detect_face(image, coords)
        if any(x < 0 for x in coords):
            return {'neutral': 0.0001, 'happy': 0.0001, 'sad': 0.0001, 'surprise': 0.0001, 'fear': 0.0001, 'disgust': 0.0001, 'angry': 0.0001}
        
        out = self.emotixv4emotion.predict(face_img)
        return out


class Pulse:
    def __init__(self) -> None:
        self.pulse_detector = PulseDetector()

    def detect_face(self, image: np.ndarray, coords) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        faces = [coords]
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            face = image[y:y+h, x:x+w]
            return face, (x, y, w, h)
        return None, None

    def predict(self, image: np.ndarray, coords: tuple = (None,None,None,None)) -> Optional[Dict[str, float]]:
        if image is None:
            return None
        # Detect and crop face
        face_img, _ = self.detect_face(image, coords)
        if face_img is []:
            return None
        
        out = self.pulse_detector.get_pulse(face_img)
        return out if out else 0.0



class FaceAnalysisApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Анализ эмоций")
        self.setGeometry(100, 100, 1280, 800)
        self.emotion_processors = {
            "EmotiX-SMALL": self.process_emotions_small,
            "EmotiX-Thermal [WIP]": self.process_emotions_small
        }
        self.current_processor = "EmotiX-SMALL"

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
            QComboBox {
                background-color: #ffffff;
                border: 2px solid #3498db;
                border-radius: 5px;
                padding: 8px;
                min-width: 200px;
                color: #2c3e50;
                font-size: 14px;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 20px;
            }
            QComboBox::down-arrow {
                image: url(down_arrow.png);
                width: 12px;
                height: 12px;
            }
            QComboBox:hover {
                border-color: #2980b9;
            }
            QComboBox QAbstractItemView {
                background-color: white;
                border: 2px solid #3498db;
                border-radius: 5px;
                selection-background-color: #3498db;
                selection-color: white;
            }
            #modelSelectionFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 15px;
                margin: 10px;
            }
            #modelSelectionLabel {
                font-size: 16px;
                font-weight: bold;
                color: #2c3e50;
                margin-bottom: 10px;
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

        # Создаем стильную панель выбора модели в правой колонке
        model_selection_frame = QFrame()
        model_selection_frame.setObjectName("modelSelectionFrame")
        model_selection_layout = QVBoxLayout(model_selection_frame)
        
        # Заголовок для выбора модели
        model_selection_label = QLabel("Выбор модели классификации")
        model_selection_label.setObjectName("modelSelectionLabel")
        model_selection_layout.addWidget(model_selection_label)
        
        # Комбобокс для выбора модели
        self.processor_combo = QComboBox()
        self.processor_combo.addItems(list(self.emotion_processors.keys()))
        self.processor_combo.currentTextChanged.connect(self.change_emotion_processor)
        model_selection_layout.addWidget(self.processor_combo)
        
        # Добавляем растягивающийся спейсер
        model_selection_layout.addStretch()
        
        pulse_frame = QFrame()
        pulse_frame.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        pulse_layout = QVBoxLayout(pulse_frame)

        pulse_title = QLabel("Пульс")
        pulse_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pulse_title.setStyleSheet("""
            font-size: 16px;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        """)
        pulse_layout.addWidget(pulse_title)

        # Current Pulse
        current_pulse_layout = QHBoxLayout()
        self.current_pulse_value = QLabel("-- уд/мин")
        current_pulse_layout.addWidget(self.current_pulse_value)
        current_pulse_widget = QWidget()
        current_pulse_widget.setLayout(current_pulse_layout)
        pulse_layout.addWidget(current_pulse_widget)

        # Стилизация значений пульса
        pulse_value_style = """
            QLabel {
                font-size: 14px;
                color: #2c3e50;
                padding: 5px;
                background-color: #f8f9fa;
                border-radius: 5px;
                min-width: 80px;
            }
        """
        self.current_pulse_value.setStyleSheet(pulse_value_style)

        # Добавляем панель выбора модели в правую колонку

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
        
        #self.stress_chart = StressChart(self, width=8, height=2)
        #charts_layout.addWidget(self.stress_chart)
        
        self.emotions_chart = EmotionsChart(self, width=8, height=2)
        charts_layout.addWidget(self.emotions_chart)      
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


        self.select_model_button = QPushButton("Выбрать модель")
        self.select_model_button.clicked.connect(self.select_model)
        controls_layout.addWidget(self.select_model_button)


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

        self.stress_button = QPushButton("Учитывать стресс в эмоциях - выключено", self)
        self.stress_button.setCheckable(True)
        self.stress_button.setChecked(False)
        self.stress_button.clicked.connect(self.toggle_stress_consideration)

        controls_layout.addWidget(self.stress_button)
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
        emotion_translations = {
            "sad": "грусть",
            "disgust": "отвращение",
            "angry": "злость",
            "neutral": "нейтральность",
            "fear": "страх",
            "surprise": "удивление",
            "happy": "радость"
        }
        
        zero_emotions = {
            "neutral": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "surprise": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "angry": 0.0
        }
        # Берем только 4 самых выраженных
        emotion_text = "Обнаруженные эмоции в кадре:\n"
        for emotion, probability in zero_emotions.items():
            translated_emotion = emotion_translations.get(emotion, emotion)
            emotion_text += f"{translated_emotion}: {probability:.2%}\n"
        # Clear emotion label
        self.emotions_label.setText(emotion_text)
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
        right_layout.addWidget(pulse_frame)


        self.timer_label = QLabel("00:00")
        self.timer_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.timer_label.setStyleSheet("font-size: 24px; color: #2c3e50;")
        controls_layout.addWidget(self.timer_label)
        #right_layout.addWidget(model_selection_frame)
        # Spacer to push controls to the top
        right_layout.addStretch()

        # Skip frames for video file
        self.frame_skip = 3

        # Initialize other variables and setup
        self.emotion_emotixsmall = EmotionModel("models_emotix_v4/emotion/checkpoint.bin")
        self.stress_detector = StressModel("models_emotix_v4/stress")
        self.pulse_predictor = Pulse()
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
        self.gaze_values = []
        self.avg_gaze = None
        self.stress_levels = []
        self.avg_stress = None
        self.consider_stress = False



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
        self.emotions_chart.clear_chart()
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
        self.emotions_chart.clear_chart()

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

        emotion_translations = {
            "sad": "грусть",
            "disgust": "отвращение",
            "angry": "злость",
            "neutral": "нейтральность",
            "fear": "страх",
            "surprise": "удивление",
            "happy": "радость"
        }
        
        zero_emotions = {
            "neutral": 0.0,
            "happy": 0.0,
            "sad": 0.0,
            "surprise": 0.0,
            "fear": 0.0,
            "disgust": 0.0,
            "angry": 0.0
        }
        emotion_text = "Обнаруженные эмоции в кадре:\n"
        for emotion, probability in zero_emotions.items():
            translated_emotion = emotion_translations.get(emotion, emotion)
            emotion_text += f"{translated_emotion}: {probability:.2%}\n"
        # Clear emotion label
        self.emotions_label.setText(emotion_text)
        self.current_pulse_value.setText("-- уд/мин")
        # Reset video label
        self.video_label.clear()

    def update_stopwatch(self):
        self.stopwatch_seconds += 1
        minutes = self.stopwatch_seconds // 60
        seconds = self.stopwatch_seconds % 60
        self.timer_label.setText(f"{minutes:02d}:{seconds:02d}")

    def process_emotions_small(self, frame, coords):
        return self.emotion_emotixsmall.predict(frame, 5, coords)
    
    def change_emotion_processor(self, processor_name):
        self.current_processor = processor_name
        return None
    
    def toggle_stress_consideration(self):
        self.consider_stress = self.stress_button.isChecked()
        if self.consider_stress:
            self.stress_button.setText("Учитывать стресс в эмоциях - включено")
        else:
            self.stress_button.setText("Учитывать стресс в эмоциях - выключено")
        return
        
    def select_model(self):
        dialog = ModelSelectionDialog(self, self.current_processor)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            selected_model = dialog.get_selected_model()
            self.change_emotion_processor(selected_model)


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
            processor_func = self.emotion_processors[self.current_processor]
            # Resize frame to 480x320
            frame = cv2.resize(frame, (480, 320))
            # Process the frame
            face_landmarks, hands_landmarks, coords = find_face_and_hands(frame, self.face_mesh, self.hands)
            
            if face_landmarks:
                face = face_landmarks.landmark
                self.face_area_size = get_face_relative_area(face)

                # Get gaze direction
                avg_gaze = get_avg_gaze(face)
                self.gaze_values.append(avg_gaze)
                if len(self.gaze_values) > 60:
                    self.gaze_values = []

                bpm_display = self.pulse_predictor.predict(frame, coords)

                # Update heart rate chart

                bpm_value = float(bpm_display)
                self.bpm_values.append(bpm_value)
                self.current_pulse_value.setText(f"{bpm_value:.0f} уд/мин")
                self.chart.update_chart(current_time, bpm_value)

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

                emotions = processor_func(frame, coords)

                # Use Stress for Emotion Detection
                if self.consider_stress:
                    current_stress = np.mean(self.stress_levels) if self.stress_levels else 0
                    emotions = self.adjust_emotions_with_stress(emotions, current_stress)
                self.emotions_chart.update_chart(current_time, emotions)

                
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
                stress_level = self.stress_detector.detect_stress(frame, face)
                stress_level_rule = self.detect_stress_rule()
                stress_level = stress_level + stress_level_rule
                stress_level = stress_level if stress_level <= 1.0 else 1.0 
                self.stress_levels.append(stress_level)
                if len(self.stress_levels) > 60:
                    self.stress_levels = self.stress_levels[-60:]
                #self.stress_chart.update_chart(current_time, stress_level * 100)
                self.update_average_stress()

            frame_with_face = self.draw_face_frame(frame, coords)
            rgb_image = cv2.cvtColor(frame_with_face, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))
            
    def draw_face_frame(self, frame, face_coords):
        x, y, w, h = face_coords
        if x is not None:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 167, 0), 1)
        return frame

    def detect_stress_rule(self):

        frame_stress = 0
        bpm_stress = 0
        lip_stress = 0
        blink_stress = 0
        hand_stress = 0
        gaze_stress = 0

        if self.bpm_values:
            avg_bpm = np.mean(self.bpm_values[-5:])
            if avg_bpm > 100:
                bpm_stress = 1.1
                frame_stress += 0.25

        lip_stress = 0
        if self.lip_ratio_values:
            avg_lip_ratio = np.mean(self.lip_ratio_values[-10:])
            if avg_lip_ratio < 0.3:
                lip_stress = 1.01
                frame_stress += 0.25

        if self.blink_count > 20:
            blink_stress = 1.05
            frame_stress += 0.20

        gaze_stress = 0
        if self.gaze_values:
            gaze_change = calculate_gaze_score(self.gaze_values)
            gaze_stress = gaze_change
            frame_stress += gaze_change

        
        #frame_stress = bpm_stress + lip_stress + blink_stress + hand_stress + gaze_stress
        """
        print("-"*30)
        print(f"Уровень стресса по сердечному ритму (BPM): {bpm_stress}")
        print(f"Уровень стресса по губам: {lip_stress}")
        print(f"Уровень стресса по морганию: {blink_stress}")
        print(f"Уровень стресса по движениям руки: {hand_stress}")
        print(f"Уровень стресса по взгляду: {gaze_stress}")
        print(f"Общий уровень стресса на кадр: {frame_stress}")
        """
        
        return frame_stress
    def adjust_emotions_with_stress(self, emotions, stress_level):
        stress_factor = stress_level 

        adjusted_emotions = emotions.copy()
        
        for emotion in ['angry', 'fear', 'sad', 'disgust', 'happy', 'surprise']:
            if emotion in adjusted_emotions:
                adjusted_emotions[emotion] *= (1 + stress_factor * 0.1)
        
        if 'neutral' in adjusted_emotions:
            adjusted_emotions['neutral'] *= (1 - stress_factor * 0.05)
        
        total = sum(adjusted_emotions.values())
        adjusted_emotions = {k: v / total for k, v in adjusted_emotions.items()}
        
        return adjusted_emotions

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