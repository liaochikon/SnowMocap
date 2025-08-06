import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QFileDialog, QVBoxLayout, QWidget
from snowvision import CameraGroup, ChessBoard, Load_Config_Json, Load_Video

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Multi-Camera Calibration Tool")
        self.setGeometry(100, 100, 400, 200)

        self.checker_info_files = []
        self.in_cal_vid_files = []
        self.ex_cal_vid_files = []

        layout = QVBoxLayout()

        self.in_cal_vid_open_button = QPushButton("Select Intrinsic Calibration Videos")
        self.in_cal_vid_open_button.clicked.connect(self.open_in_cal_vid_dialog)
        layout.addWidget(self.in_cal_vid_open_button)

        self.in_cal_vid_path_label = QLabel("No intrinsic calibration videos selected")
        layout.addWidget(self.in_cal_vid_path_label)

        self.checker_info_open_button = QPushButton("Select Intrinsic Calibration Checker Info")
        self.checker_info_open_button.clicked.connect(self.open_checker_info)
        layout.addWidget(self.checker_info_open_button)

        self.checker_info_path_label = QLabel("No intrinsic calibration checker info selected")
        layout.addWidget(self.checker_info_path_label)

        self.ex_cal_vid_open_button = QPushButton("Select Extrinsic Calibration Videos")
        self.ex_cal_vid_open_button.clicked.connect(self.open_ex_cal_vid_dialog)
        layout.addWidget(self.ex_cal_vid_open_button)

        self.ex_cal_vid_path_label = QLabel("No extrinsic calibration videos selected")
        layout.addWidget(self.ex_cal_vid_path_label)

        self.cal_start_button = QPushButton("Start Calibration")
        self.cal_start_button.clicked.connect(self.calibration)
        layout.addWidget(self.cal_start_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_checker_info(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        file_dialog.setNameFilter("JSON Files (*.json)")
        
        if file_dialog.exec_():
            self.checker_info_files = file_dialog.selectedFiles()
            if self.checker_info_files:
                self.checker_info_path_label.setText(self.checker_info_files[0])
        

    def open_in_cal_vid_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("MP4 Files (*.mp4);;AVI Files (*.avi)")
        
        if file_dialog.exec_():
            self.in_cal_vid_files = file_dialog.selectedFiles()
            if self.in_cal_vid_files:
                selected_files_str = ""
                for f in self.in_cal_vid_files:
                    selected_files_str += f + ", \n"
                self.in_cal_vid_path_label.setText(selected_files_str)

    def open_ex_cal_vid_dialog(self):
        file_dialog = QFileDialog(self)
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        file_dialog.setNameFilter("MP4 Files (*.mp4);;AVI Files (*.avi)")
        
        if file_dialog.exec_():
            self.ex_cal_vid_files = file_dialog.selectedFiles()
            if self.ex_cal_vid_files:
                selected_files_str = ""
                for f in self.ex_cal_vid_files:
                    selected_files_str += f + ", \n"
                self.ex_cal_vid_path_label.setText(selected_files_str)    
    
    def calibration(self):
        if len(self.checker_info_files) == 0:
            print("ERROR : no checker info file")
            return
        if len(self.in_cal_vid_files) == 0:
            print("ERROR : no intrinsic calibration video file")
            return
        # if len(self.ex_cal_vid_files) == 0:
        #     print("ERROR : no extrinsic calibration video file")
        #     return
        # if (len(self.in_cal_vid_files) != len(self.ex_cal_vid_files)):
        #     print("ERROR : intrinsic & extrinsic calibration video number not match")
        #     return

        in_cal_vid_idx = []
        in_cal_vid_res = []
        for in_cal_vid in self.in_cal_vid_files:
            in_cal_vid_id = int(in_cal_vid[in_cal_vid.find('_i') + 2])
            in_cal_vid_idx.append(in_cal_vid_id)

            width = int(in_cal_vid[in_cal_vid.find('_r') + 2:in_cal_vid.find('x')])
            height = int(in_cal_vid[in_cal_vid.find('x') + 1:in_cal_vid.find('_d')])
            in_cal_vid_res.append([width, height])

        checker_info_json = Load_Config_Json(self.checker_info_files[0])
        chessboard = ChessBoard(checker_info_json['height'] - 1, checker_info_json['width'] - 1, checker_info_json['square_size_mm'] / 1000)
        video_dict_list = [Load_Video(video_name) for video_name in self.in_cal_vid_files]
        video_cap_list = [video_dict['cap'] for video_dict in video_dict_list]
        video_length_list = [video_dict['length'] for video_dict in video_dict_list]

        cameragroup = CameraGroup(cap_ids=in_cal_vid_idx, resolutions=in_cal_vid_res)
        cameragroup.intrinsic_calibrate_video(video_cap_list, video_length_list, chessboard, show_vid=True)
        cameragroup.save_camera_group_info(f"camera_n{cameragroup.camera_num}_info.json")

        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())