import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                           QPushButton, QComboBox, QLabel, QGridLayout)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from datetime import datetime

class CameraRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Recorder")
        self.setGeometry(100, 100, 800, 300)
        
        # Initialize variables
        self.cameras = []
        self.active_cameras = []
        self.is_recording = False
        self.video_writers = []
        self.resolution = (1280, 720)
        self.fps = 30
        
        # Create main widget and layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout(self.main_widget)
        
        # Camera preview layout
        self.preview_layout = QGridLayout()
        self.layout.addLayout(self.preview_layout)
        
        # Control panel
        control_widget = QWidget()
        control_layout = QHBoxLayout(control_widget)
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.addItem("All Cameras")
        control_layout.addWidget(QLabel("Select Camera:"))
        control_layout.addWidget(self.camera_combo)
        
        # Control buttons
        self.detect_button = QPushButton("Detect Cameras")
        self.detect_button.clicked.connect(self.detect_cameras)
        control_layout.addWidget(self.detect_button)
        
        self.record_button = QPushButton("Start Recording")
        self.record_button.clicked.connect(self.toggle_recording)
        control_layout.addWidget(self.record_button)
        
        self.layout.addWidget(control_widget)
        
        # Preview labels
        self.preview_labels = []
        
        # Timer for updating preview
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        
        # Detect cameras on startup
        self.detect_cameras()
        
    def detect_cameras(self):
        """Detect available cameras"""
        self.cameras = []
        self.camera_combo.clear()
        self.camera_combo.addItem("All Cameras")
        self.cameras_to_record = []
        
        # Clear existing preview labels
        for label in self.preview_labels:
            label.deleteLater()
        self.preview_labels = []
        
        for index in range(10):
            cap = cv2.VideoCapture(index)
            if not cap.isOpened():
                continue
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            self.active_cameras.append(cap)
            self.cameras.append(index)
            self.camera_combo.addItem(f"Camera {index}")
            label = QLabel()
            label.setFixedSize(400, 225)
            self.preview_labels.append(label)
            self.preview_layout.addWidget(label, 0, len(self.preview_labels)-1 % 2)
        
        if not self.cameras:
            self.detect_button.setText("No Cameras Found")
        else:
            self.detect_button.setText(f"Found {len(self.cameras)} Camera(s)")
            self.timer.start(int(1000 / self.fps / 4))
            
    def update_frames(self):
        for i, (cap, label) in enumerate(zip(self.active_cameras, self.preview_labels)):
            ret, frame = cap.read()
            if ret:
                frame_preview = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_preview = cv2.resize(frame_preview, (400, 225))
                h, w, ch = frame_preview.shape
                bytes_per_line = ch * w
                image = QImage(frame_preview.data, w, h, bytes_per_line, QImage.Format_RGB888)
                label.setPixmap(QPixmap.fromImage(image))
                
                if self.is_recording:
                    if len(self.cameras_to_record) == 1 and i == self.cameras_to_record[0]:
                        self.video_writers[0].write(frame)
                    if len(self.cameras_to_record) > 1:
                        self.video_writers[i].write(frame)
    
    def toggle_recording(self):
        """Start or stop recording"""
        if not self.is_recording:
            self.is_recording = True
            self.record_button.setText("Stop Recording")
            
            self.video_writers = []
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            selected_camera = self.camera_combo.currentText()
            self.camera_combo.setEnabled(False)
            self.detect_button.setEnabled(False)
            
            if selected_camera == "All Cameras":
                self.cameras_to_record = self.cameras
            else:
                camera_idx = int(selected_camera.split()[-1])
                self.cameras_to_record = [camera_idx]
            
            for idx in self.cameras_to_record:
                filename = f"camera_i{idx}_r{self.resolution[0]}x{self.resolution[1]}_d{timestamp}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(filename, fourcc, self.fps, self.resolution)
                self.video_writers.append(writer)
                
        else:
            self.is_recording = False
            self.record_button.setText("Start Recording")
            self.camera_combo.setEnabled(True)
            self.detect_button.setEnabled(True)

            # Release video writers
            for writer in self.video_writers:
                writer.release()
            self.video_writers = []
    
    def closeEvent(self, event):
        """Clean up on window close"""
        for cap in self.active_cameras:
            cap.release()
        for writer in self.video_writers:
            writer.release()
        self.timer.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CameraRecorder()
    window.show()
    sys.exit(app.exec_())