import json
import pickle
from random import sample
import socket
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import cv2
import numpy as np

class ChessBoard:
    height = 9
    width = 6
    square_size = 0.095
    objp = np.zeros((width * height, 3), np.float32)
    objp[:, :2] = square_size * np.mgrid[0:height, 0:width].T.reshape(-1, 2)
    criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.001)

def Get_T0():
    R0 = np.eye(3)
    t0 = np.array([[0], [0], [0]])
    T = np.hstack((R0, t0))
    T = np.vstack((T, [0, 0, 0, 1]))
    return T

isCapturing = False
Ports = [6666 , 6667, 6668, 6669]
CurrentCamera = 0
Mode = 0 #0 for K, 1 for Rt 
iter_num = 70

chessboard = ChessBoard()

total_corners_K = []
draw_total_corners_K = []
total_corners_Rt = []
draw_total_corners_Rt = []
sizes = []

K_Result = []
D_Result = []
T_Stereo_Result = []
T_Multi_Result = []

for i in range(len(Ports)):
    total_corners_K.append([])
    draw_total_corners_K.append(np.array([]))
    total_corners_Rt.append([[], []])
    draw_total_corners_Rt.append([np.array([]), np.array([])])
    sizes.append([0, 0])

    K_Result.append([])
    D_Result.append([])
    T_Stereo_Result.append([])
    T_Multi_Result.append([])
T_Multi_Result[0] = Get_T0()

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("SnowCal")
        #self.setGeometry(0, 0, 1280, 720)

        self.filename = "snow_output"

        self.VBL = QGridLayout()

        self.FeedLabel1 = QLabel(self)
        self.VBL.addWidget(self.FeedLabel1, 0,0)

        self.Worker1 = Worker1("192.168.50.237")
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.Image1UpdateSlot)
        self.FeedLabel1.resize(640,360)
        
        v_layout = self.ControlPanel()
        self.VBL.addItem(v_layout, 1, 0)

        self.setLayout(self.VBL)
        self.show()

    def ControlPanel(self):
        BTN_h = 25
        BTN_w = 100

        v_layout = QVBoxLayout()

        CaptureCornerBTN = QPushButton("CaptureCorner")
        CaptureCornerBTN.setFixedSize(QSize(BTN_w, BTN_h))
        CaptureCornerBTN.clicked.connect(self.CaptureCorner)
        v_layout.addWidget(CaptureCornerBTN)

        ChangeCameraBTN = QPushButton("ChangeCamera")
        ChangeCameraBTN.setFixedSize(QSize(BTN_w, BTN_h))
        ChangeCameraBTN.clicked.connect(self.ChangeCamera)
        v_layout.addWidget(ChangeCameraBTN)

        ChangeModeBTN = QPushButton("ChangeMode")
        ChangeModeBTN.setFixedSize(QSize(BTN_w, BTN_h))
        ChangeModeBTN.clicked.connect(self.ChangeMode)
        v_layout.addWidget(ChangeModeBTN)

        CalibrateBTN = QPushButton("Calibrate")
        CalibrateBTN.setFixedSize(QSize(BTN_w, BTN_h))
        CalibrateBTN.clicked.connect(self.Calibrate)
        v_layout.addWidget(CalibrateBTN)

        MultiCalibrateBTN = QPushButton("MultiCalibrate")
        MultiCalibrateBTN.setFixedSize(QSize(BTN_w, BTN_h))
        MultiCalibrateBTN.clicked.connect(self.MultiCalibrate)
        v_layout.addWidget(MultiCalibrateBTN)

        #CalibrateFloorBTN = QPushButton("CalibrateFloor")
        #CalibrateFloorBTN.setFixedSize(QSize(BTN_w, BTN_h))
        #CalibrateFloorBTN.clicked.connect(self.CalibrateFloor)
        #v_layout.addWidget(CalibrateFloorBTN)

        OutputDataBTN = QPushButton("OutputData")
        OutputDataBTN.setFixedSize(QSize(BTN_w, BTN_h))
        OutputDataBTN.clicked.connect(self.OutputData)
        v_layout.addWidget(OutputDataBTN)

        LoadOutputDataBTN = QPushButton("LoadOutputData")
        LoadOutputDataBTN.setFixedSize(QSize(BTN_w, BTN_h))
        LoadOutputDataBTN.clicked.connect(self.LoadOutputData)
        v_layout.addWidget(LoadOutputDataBTN)

        RecoverBackupBTN = QPushButton("RecoverBackup")
        RecoverBackupBTN.setFixedSize(QSize(BTN_w, BTN_h))
        RecoverBackupBTN.clicked.connect(self.RecoverBackup)
        v_layout.addWidget(RecoverBackupBTN)

        SaveBackupBTN = QPushButton("SaveBackup")
        SaveBackupBTN.setFixedSize(QSize(BTN_w, BTN_h))
        SaveBackupBTN.clicked.connect(self.SaveBackup)
        v_layout.addWidget(SaveBackupBTN)

        ClearImagePointsBTN = QPushButton("ClearImagePoints")
        ClearImagePointsBTN.setFixedSize(QSize(BTN_w, BTN_h))
        ClearImagePointsBTN.clicked.connect(self.ClearImagePoints)
        v_layout.addWidget(ClearImagePointsBTN)

        return v_layout
    
    def Image1UpdateSlot(self, Image):
        self.FeedLabel1.setPixmap(QPixmap.fromImage(Image))

    def CalibrateFloor(self):
        global isCapturing
        global T_Multi_Result

        isCapturing = False
        x = self.Worker1.LanCameras[0].recvfrom(1000000)
        data = x[0]
        data = pickle.loads(data)
        Image = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, 1, 0]], np.float)
        gray = cv2.filter2D(gray, -1, kernel=kernel)
        sizes[CurrentCamera] = [Image.shape[1], Image.shape[0]]
        ret, corners = cv2.findChessboardCorners(gray, (chessboard.height, chessboard.width), None)
        if ret:
            r_norm = np.array([])
            for i in range(50):
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), chessboard.criteria)
                _, _, _, r, t = cv2.calibrateCamera([chessboard.objp], [corners_sub], sizes[0], None, None)
                r_norm = np.append(r_norm, r[0] / np.linalg.norm(r[0]))
            r_norm = r_norm.reshape(50, 3)
            r_norm = np.mean(r_norm, axis=0)
            R = cv2.Rodrigues(r_norm)[0]
            T = np.hstack((R, t[0]))
            T = np.vstack((T, [0, 0, 0, 1]))

        
    def SaveBackup(self):
        print("Save imagepoints as bonfire.json")
        total_corners_K_out = []
        draw_total_corners_K_out = []
        total_corners_Rt_out = []
        draw_total_corners_Rt_out = []
        for i in range(len(Ports)):
            total_corners_K_out.append([])
            draw_total_corners_K_out.append([])
            total_corners_Rt_out.append([[], []])
            draw_total_corners_Rt_out.append([[], []])
        
        with open("bonfire" + ".json" , "w") as outfile:
            for i, t in enumerate(total_corners_K):
                for tt in t:
                    total_corners_K_out[i].append(tt.tolist())
            for i, t in enumerate(total_corners_Rt):
                for j, tt in enumerate(t):
                    for ttt in tt:
                        total_corners_Rt_out[i][j].append(ttt.tolist())
            draw_total_corners_K_out = [i.tolist() for i in draw_total_corners_K]
            draw_total_corners_Rt_out = [[i[0].tolist(), i[1].tolist()] for i in draw_total_corners_Rt]
            jsonStr = json.dumps({'sizes':sizes, 'total_corners_K':total_corners_K_out, 'draw_total_corners_K':draw_total_corners_K_out, 'total_corners_Rt':total_corners_Rt_out, 'draw_total_corners_Rt':draw_total_corners_Rt_out})
            outfile.write(jsonStr)

    def RecoverBackup(self):
        global sizes, total_corners_K, draw_total_corners_K, total_corners_Rt, draw_total_corners_Rt
        total_corners_K = []
        draw_total_corners_K = []
        total_corners_Rt = []
        draw_total_corners_Rt = []
        sizes = []

        for i in range(len(Ports)):
            total_corners_K.append([])
            draw_total_corners_K.append(np.array([]))
            total_corners_Rt.append([[], []])
            draw_total_corners_Rt.append([np.array([]), np.array([])])
            sizes.append([0, 0])

        with open("bonfire" + ".json" , "r") as f:
            data = json.load(f)
            sizes = data['sizes']
            for i, t in enumerate(data['total_corners_K']):
                for tt in t:
                    total_corners_K[i].append(np.array(tt, dtype=np.float32))
            for i, t in enumerate(data['total_corners_Rt']):
                for j, tt in enumerate(t):
                    for ttt in tt:
                        total_corners_Rt[i][j].append(np.array(ttt, dtype=np.float32))
            for i, t in enumerate(data['draw_total_corners_K']):
                draw_total_corners_K[i] = np.array(t, dtype=np.float32)
            for i, t in enumerate(data['draw_total_corners_Rt']):
                for j, tt in enumerate(t):
                    draw_total_corners_Rt[i][j] = np.array(tt, dtype=np.float32)

    def CaptureCorner(self):
        global isCapturing
        if isCapturing:
            isCapturing = False 
            return
        isCapturing = True
        return

    def ChangeCamera(self):
        global CurrentCamera, isCapturing
        isCapturing = False 
        if CurrentCamera < len(Ports) - 1:
            CurrentCamera += 1
            return
        CurrentCamera = 0
        return

    def ChangeMode(self):
        global Mode, isCapturing
        isCapturing = False 
        if Mode == 0:
            Mode = 1
            return
        Mode = 0
        return

    def Calibrate(self):
        global isCapturing
        isCapturing = False 
        if Mode == 0:
            K_Result[CurrentCamera], D_Result[CurrentCamera] = self.Single_Calibrate(CurrentCamera, iter_num, sizes[CurrentCamera])
            print("Calibrating " + str(CurrentCamera) + " K")
            return
        if Mode == 1:
            T_Stereo_Result[CurrentCamera] = self.Stereo_Calibrate(CurrentCamera, iter_num, sizes[CurrentCamera])
            print("Calibrating " + str(CurrentCamera) + " Rt")
            return

    def ClearImagePoints(self):
        global total_corners_K, draw_total_corners_K, total_corners_Rt, draw_total_corners_Rt, isCapturing
        isCapturing = False 
        
        if Mode == 0:
            total_corners_K[CurrentCamera] = []
            draw_total_corners_K[CurrentCamera] = np.array([])
            sizes[CurrentCamera] = [0, 0]
            return
        if Mode == 1:
            total_corners_Rt[CurrentCamera] = [[], []]
            draw_total_corners_Rt[CurrentCamera] = [np.array([]), np.array([])]
            sizes[CurrentCamera] = [0, 0]
            return

    def Single_Calibrate(self, current_cam, limit_num, img_size):
        if len(total_corners_K[current_cam]) < limit_num:
            return [], []

        newcameramtx = []
        gap = len(total_corners_K[current_cam]) // limit_num
        Single_objectPoints_temp = []
        Single_imagePoints_temp = []
        l = len(total_corners_K[current_cam])
        pad = 1
        for i in range(limit_num):
            index = i * gap
            if index >= l:
                if l % gap == 1:
                    index = i * gap - l
                else:
                    index = i * gap - l + pad
            Single_objectPoints_temp.append(chessboard.objp)
            Single_imagePoints_temp.append(total_corners_K[current_cam][index])

        ret, mtx, dist, _, _ = cv2.calibrateCamera(Single_objectPoints_temp, Single_imagePoints_temp, img_size, None, None)
        if ret:
            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx, dist, img_size, 1, img_size)
            return newcameramtx, dist
        return [], []

    def Stereo_Calibrate(self, current_cam, limit_num, img_size):
        if len(total_corners_Rt[current_cam][0]) < limit_num:
            return []

        Stereo_objectPoints = []
        Stereo_imagePoints_l = []
        Stereo_imagePoints_r = []
        random_index = sample(range(len(total_corners_Rt[current_cam][0])), limit_num)
        for i in random_index:
            Stereo_objectPoints.append(chessboard.objp)
            Stereo_imagePoints_l.append(total_corners_Rt[current_cam][0][i])
            Stereo_imagePoints_r.append(total_corners_Rt[current_cam][1][i])

        ret1, mtx_l, dist_l, rvecs1, tvecs1 = cv2.calibrateCamera(Stereo_objectPoints, Stereo_imagePoints_l, img_size, None, None)
        ret2, mtx_r, dist_r, rvecs2, tvecs2 = cv2.calibrateCamera(Stereo_objectPoints, Stereo_imagePoints_r, img_size, None, None)
    
        ret_s, mtx_s1, dist_s1, mtx_s2, dist_s2, rvecs_s, tvecs_s, E, F = cv2.stereoCalibrate(Stereo_objectPoints,
                                                                                              Stereo_imagePoints_l,
                                                                                              Stereo_imagePoints_r,
                                                                                              mtx_l, dist_l, mtx_r, dist_r, img_size)
        T = np.hstack((rvecs_s, tvecs_s))
        T = np.linalg.inv(np.vstack((T, [0, 0, 0, 1])))
        return T

    def MultiCalibrate(self):#not a circle yet
        global T_Multi_Result
        T_Multi_Result_temp0 = T_Multi_Result.copy()
        
        for i in range(1 ,len(Ports)):
            T_Multi_Result_temp0[i] = np.matmul(T_Stereo_Result[i - 1], T_Multi_Result_temp0[i - 1])

        T_Multi_Result[1] = T_Multi_Result_temp0[1]
        T_Multi_Result[2] = T_Multi_Result_temp0[2]
        T_Multi_Result[3] = T_Multi_Result_temp0[3]
        return

    def OutputData(self):
        with open(self.filename + '.txt', 'w') as f:
            usable_cam_num = len(Ports)
            f.write('Usable_Cam_Num=' + str(usable_cam_num) + '\n')
            for num in range(usable_cam_num):
                f.write('Cam_' + str(num) + ':\n')
                K_temp = K_Result[num].reshape((9))
                f.write('K=')
                for k in K_temp:
                    f.write(str(float(k)) + ',')
                f.write('\n')
                Rt_temp = T_Multi_Result[num][:3].reshape((12))
                f.write('Rt=')
                for Rt in Rt_temp:
                    f.write(str(float(Rt)) + ',')
                f.write('\n')
                D_temp = D_Result[num][0]
                f.write('D=')
                for D in D_temp:
                    f.write(str(float(D)) + ',')
                print(K_temp)
                print(Rt_temp)
                print(D_temp)
                f.write('\n')
    
    def Read_Array(self, line, shape):
        K = np.array([])
        K_str = line.split('=')[1].split(',')
        for i in range(int(shape[0] * shape[1])):
            k_temp = float(K_str[i])
            K = np.append(K, k_temp)
        K = K.reshape(shape)
        return K

    def LoadOutputData(self):
        global K_Result, T_Multi_Result, D_Result
        K_Result = []
        D_Result = []
        T_Multi_Result = []
        with open(self.filename + '.txt', 'r') as f:
            lines = f.readlines()
            camera_count = int(lines[0].split('=')[1])
            print("Usable_Cam_Num：",camera_count)
            for i in range(camera_count):
                index = int(1 + i * 4)
                K = self.Read_Array(lines[index + 1], (3, 3))
                print(K)
                K_Result.append(K)
                Rt = self.Read_Array(lines[index + 2], (3, 4))
                T = np.linalg.inv(np.vstack((Rt, [0, 0, 0, 1])))
                print(T)
                T_Multi_Result.append(T)
                D = self.Read_Array(lines[index + 3], (1, 5))
                print(D)
                D_Result.append(D)
        
                    

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)

    def __init__(self, ip):
        super(Worker1, self).__init__()
        self.ip = ip

    def run(self):
        self.ThreadActive = True

        self.LanCameras = []
        for p in Ports:
            soc=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
            soc.bind((self.ip, p))
            self.LanCameras.append(soc)      
            
        while self.ThreadActive:
            if Mode == 0:
                self.K_Mode()
            if Mode == 1:
                self.Rt_Mode()

    def K_Mode(self):
        global total_corners_K, draw_total_corners_K, sizes
        x = self.LanCameras[CurrentCamera].recvfrom(1000000)
        data = x[0]
        data = pickle.loads(data)
        Image = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, 1, 0]], np.float)
        gray = cv2.filter2D(gray, -1, kernel=kernel)
        sizes[CurrentCamera] = [Image.shape[1], Image.shape[0]]

        isCapturing_info = ", not capturing corners"
        if isCapturing:
            isCapturing_info = ", capturing corners"
            ret, corners = cv2.findChessboardCorners(gray, (chessboard.height, chessboard.width), None)
            if ret:
                corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), chessboard.criteria)
                total_corners_K[CurrentCamera].append(corners_sub)
                draw_total_corners_K[CurrentCamera] = np.append(draw_total_corners_K[CurrentCamera], [corners_sub[0][0], corners_sub[8][0], corners_sub[53][0], corners_sub[45][0]])
                draw_total_corners_K[CurrentCamera] = draw_total_corners_K[CurrentCamera].reshape((len(draw_total_corners_K[CurrentCamera]) // 2 // 4, 4, 2))
        cv2.putText(Image, "mode K", (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        cv2.putText(Image, "cam " + str(CurrentCamera) + ", corners " + str(len(draw_total_corners_K[CurrentCamera])) + isCapturing_info, (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
        
        for i, board in enumerate(draw_total_corners_K[CurrentCamera]):
            cv2.line(Image,(int(board[0][0]), int(board[0][1])),(int(board[1][0]), int(board[1][1])),(0,255,255),2)
            cv2.line(Image,(int(board[1][0]), int(board[1][1])),(int(board[2][0]), int(board[2][1])),(0,255,255),2)
            cv2.line(Image,(int(board[2][0]), int(board[2][1])),(int(board[3][0]), int(board[3][1])),(0,255,255),2)
            cv2.line(Image,(int(board[3][0]), int(board[3][1])),(int(board[0][0]), int(board[0][1])),(0,255,255),2)
            cv2.line(Image,(int(board[0][0]), int(board[0][1])),(int(board[2][0]), int(board[2][1])),(0,255,255),2)
            cv2.line(Image,(int(board[1][0]), int(board[1][1])),(int(board[3][0]), int(board[3][1])),(0,255,255),2)

        ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(640, 360, Qt.KeepAspectRatio)
        self.ImageUpdate.emit(Pic)

    def Rt_Mode(self):
        global total_corners_Rt, draw_total_corners_Rt, sizes
        board_cam_vis_count = 0
        Images = []
        total_corners_Rt_temp = [[], []]
        draw_total_corners_Rt_temp = [np.array([]), np.array([])]

        for i in range(2):
            cam_index = CurrentCamera + i
            if cam_index > len(Ports) - 1:
                cam_index = 0
            x = self.LanCameras[cam_index].recvfrom(1000000)
            data = x[0]
            data = pickle.loads(data)
            Image = cv2.cvtColor(cv2.imdecode(data, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
            kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, 1, 0]], np.float)
            gray = cv2.filter2D(gray, -1, kernel=kernel)
            sizes[cam_index] = [Image.shape[1], Image.shape[0]]
            
            isCapturing_info = ", not capturing corners"
            if isCapturing:
                isCapturing_info = ", capturing corners"
                ret, corners = cv2.findChessboardCorners(gray, (chessboard.height, chessboard.width), None)
                if ret:
                    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), chessboard.criteria)
                    total_corners_Rt_temp[i] = corners_sub
                    draw_total_corners_Rt_temp[i] = [corners_sub[0][0], corners_sub[8][0], corners_sub[53][0], corners_sub[45][0]]
                    board_cam_vis_count += 1

            cv2.putText(Image, "mode Rt", (10, 40), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)
            cv2.putText(Image, "cam " + str(cam_index) + ", corners " + str(len(draw_total_corners_Rt[CurrentCamera][0])) + isCapturing_info, (10, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

            for i, board in enumerate(draw_total_corners_Rt[CurrentCamera][i]):
                cv2.line(Image,(int(board[0][0]), int(board[0][1])),(int(board[1][0]), int(board[1][1])),(0,255,255),2)
                cv2.line(Image,(int(board[1][0]), int(board[1][1])),(int(board[2][0]), int(board[2][1])),(0,255,255),2)
                cv2.line(Image,(int(board[2][0]), int(board[2][1])),(int(board[3][0]), int(board[3][1])),(0,255,255),2)
                cv2.line(Image,(int(board[3][0]), int(board[3][1])),(int(board[0][0]), int(board[0][1])),(0,255,255),2)
                cv2.line(Image,(int(board[0][0]), int(board[0][1])),(int(board[2][0]), int(board[2][1])),(0,255,255),2)
                cv2.line(Image,(int(board[1][0]), int(board[1][1])),(int(board[3][0]), int(board[3][1])),(0,255,255),2)

            Images.append(Image)
        
        if board_cam_vis_count == 2:
            for i in range(2):
                total_corners_Rt[CurrentCamera][i].append(total_corners_Rt_temp[i])
                draw_total_corners_Rt[CurrentCamera][i] = np.append(draw_total_corners_Rt[CurrentCamera][i], draw_total_corners_Rt_temp[i])
                draw_total_corners_Rt[CurrentCamera][i] = draw_total_corners_Rt[CurrentCamera][i].reshape((len(draw_total_corners_Rt[CurrentCamera][i]) // 2 // 4, 4, 2))

        Image = cv2.hconcat(Images)
        ConvertToQtFormat = QImage(Image.data, Image.shape[1], Image.shape[0], QImage.Format_RGB888)
        Pic = ConvertToQtFormat.scaled(1280, 360, Qt.KeepAspectRatio)
        self.ImageUpdate.emit(Pic)

    def stop(self):
        self.ThreadActive = False
        self.quit()

if __name__ == "__main__":
    App = QApplication(sys.argv)
    Root = MainWindow()
    sys.exit(App.exec())