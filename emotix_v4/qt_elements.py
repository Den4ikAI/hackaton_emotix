
from PyQt6.QtWidgets import (
    QVBoxLayout, QHBoxLayout,
    QPushButton, QProgressBar, QDialog, QRadioButton, QLineEdit, QFileDialog, QComboBox
)
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, 
    QPushButton, QComboBox, QLineEdit, QFileDialog, QButtonGroup
)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter 

import cv2


class EmotionsChart(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(EmotionsChart, self).__init__(fig)
        self.setParent(parent)
        
        # Инициализируем базовые списки
        self.times = []
        self.emotions_values = {}
        self.lines = {}
        
        self.initialized = False

        self.axes.set_ylim(0, 1)
        self.axes.yaxis.set_major_formatter(PercentFormatter(1.0))
        self.axes.set_xlim(0, 60)
        self.axes.set_facecolor('#f0f0f0')
        self.axes.set_xlabel(" ", fontsize=10)
        self.axes.set_ylabel("Вероятность", fontsize=10)
        self.axes.grid(True, linestyle='--', alpha=0.7)
        
        fig.tight_layout()
        #self.initialize_lines({'neutral': 0.0001, 'happy': 0.0001, 'sad': 0.0001, 'surprise': 0.0001, 'fear': 0.0001, 'disgust': 0.0001, 'angry': 0.0001})

    def initialize_lines(self, emotions_dict):
        self.emotions_colors = {
            'neutral': '#808080',    # серый
            'happy': '#FFD700',      # золотой
            'sad': '#4682B4',        # стальной синий
            'surprise': '#9932CC',    # темная орхидея
            'fear': '#800000',       # темно-красный
            'disgust': '#006400',    # темно-зеленый
            'fear': '#FF4500'       # оранжево-красный
        }
        emotion_translations = {
            "sad": "грусть",
            "disgust": "отвращение",
            "angry": "злость",
            "neutral": "нейтральность",
            "fear": "страх",
            "surprise": "удивление",
            "happy": "радость"
        }
        for emotion in emotions_dict.keys():
            self.emotions_values[emotion] = []
            line, = self.axes.plot([], [], 
                                 color=self.emotions_colors.get(emotion, '#000000'),
                                 label=emotion_translations[emotion])
            self.lines[emotion] = line
        
        self.axes.legend(loc='lower center', bbox_to_anchor=(0.5, -0.58),
                         ncol=len(emotions_dict), fontsize='small', frameon=False)
        self.initialized = True

    def update_chart(self, new_time, emotions_dict):

        if not self.initialized:
            self.initialize_lines(emotions_dict)

        self.times.append(new_time)

        for emotion in emotions_dict:
            self.emotions_values[emotion].append(emotions_dict[emotion])
        
        while self.times and self.times[0] < new_time - 60:
            self.times.pop(0)
            for emotion in emotions_dict:
                self.emotions_values[emotion].pop(0)
        
        for emotion in emotions_dict:
            self.lines[emotion].set_data(self.times, self.emotions_values[emotion])
        
        self.axes.set_xlim(max(0, new_time - 60), max(60, new_time))
        self.draw()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def clear_chart(self):
        self.times = []
        for emotion in self.emotions_values:
            self.emotions_values[emotion] = []
            self.lines[emotion].set_data([], [])

        self.axes.set_xlim(0, 60)

        # Перерисовываем график
        self.draw()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

        # Сбрасываем флаг инициализации, чтобы при следующем обновлении линии были переинициализированы
        #self.initialized = False

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
    def __init__(self, parent=None, width=5, height=4, dpi=75):
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
        
class ModelSelectionDialog(QDialog):
    def __init__(self, parent=None, current_model=None):
        super().__init__(parent)
        self.setWindowTitle("Выбор модели")
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

        self.model_group = QButtonGroup()
        for i, model_name in enumerate(["EmotiX-SMALL", "EmotiX-Thermal [WIP]"]):
            radio = QRadioButton(model_name)
            self.model_group.addButton(radio, i)
            layout.addWidget(radio)
            if model_name == current_model:
                radio.setChecked(True)

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

    def get_selected_model(self):
        return self.model_group.checkedButton().text()