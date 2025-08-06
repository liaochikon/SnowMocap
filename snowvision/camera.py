from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import numpy as np
import cv2
import json
from snowvision.triangulation import *

class ChessBoard:
    def __init__(self, height = 9, width = 6, square_size = 0.095):
        self.height = height
        self.width = width
        self.square_size = square_size
        self.objp = np.zeros((self.width * self.height, 3), np.float32)
        self.objp[:, :2] = square_size * np.mgrid[0: self.height, 0: self.width].T.reshape(-1, 2)
        self.criteria = (cv2.TERM_CRITERIA_EPS, 30, 0.001)

class Camera:
    def __init__(self, cap_id = 0, frame_width = 1280, frame_height = 720, camera_info_path = None, camera_info_dict = None):
        self.cap_id = cap_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.K = np.zeros((3, 3))
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))
        self.D = np.zeros((1, 5))

        ##CameraGroup only lists##
        self.points = []
        self.point_rays = []
        self.hrnet_points = []
        self.hrnet_point_rays = []
        self.hrnet_point_score = []
        ##CameraGroup only lists##
        
        if camera_info_path != None or camera_info_dict != None:
            if camera_info_path != None:
                with open(camera_info_path, "r") as readfile:
                    camera_info_dict = json.load(readfile)
            self.cap_id = camera_info_dict["cap_id"]
            self.frame_width = camera_info_dict["frame_width"]
            self.frame_height = camera_info_dict["frame_height"]
            self.K = np.array(camera_info_dict["K"])
            self.R = np.array(camera_info_dict["R"])
            self.t = np.array(camera_info_dict["t"])
            self.D = np.array(camera_info_dict["D"])

    def camera_info_dict(self):
        camera_info_dict = {"cap_id" : self.cap_id,
                            "frame_width" : self.frame_width,
                            "frame_height" : self.frame_height,
                            "K": self.K.tolist(),
                            "R": self.R.tolist(),
                            "t": self.t.tolist(),
                            "D": self.D.tolist()}
        return camera_info_dict
        
    def save_camera_info(self, camera_info_path):
        camera_info_dict = self.camera_info_dict()
        with open(camera_info_path, "w") as outfile:
            outfile.write(json.dumps(camera_info_dict))

    def capture_init(self):
        self.cap = cv2.VideoCapture(self.cap_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

    def snapshot(self, save_path = None):
        _, frame = self.cap.read()
        if save_path != None:
            cv2.imwrite(save_path, frame)
        
        return frame
    
    def record_init(self, video_name, video_format = ".avi", fps = 30.0):
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(video_name + video_format, self.fourcc, fps, (self.frame_width, self.frame_height))

    def recording(self, image):
        self.out.write(image)

    def record_end(self):
        self.out.release()
        
    def intrinsic_calibrate_images(self, images, chessboard : ChessBoard, sample_num = 40):
        objpoints = []
        imgpoints = [] 
        for i, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (chessboard.height, chessboard.width), None)
            if ret == False:
                continue  
            print("Get chessboard at image " + str(i))
            corners_subpix = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), chessboard.criteria)
            objpoints.append(chessboard.objp)
            imgpoints.append(corners_subpix)

        sample_idx = np.arange(0, len(imgpoints), len(imgpoints) // sample_num, dtype=int).tolist()
        objpoints = np.array(objpoints)[sample_idx]
        imgpoints = np.array(imgpoints)[sample_idx]
        
        print("Calibrating camera intrinsic...")
        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, images[0].shape[::-1], None, None)
        self.K = mtx
        self.D = dist

        return mtx, dist
    
    def intrinsic_calibrate_video(self, cap, length, chessboard : ChessBoard, sample_num = 40):
        objpoints = []
        imgpoints = [] 

        for i in range(length):
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (chessboard.height, chessboard.width), None)
            if ret == False:
                continue  
            print("Get chessboard at frame " + str(i))
            corners_subpix = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), chessboard.criteria)
            objpoints.append(chessboard.objp)
            imgpoints.append(corners_subpix)

        sample_idx = np.arange(0, len(imgpoints), len(imgpoints) // sample_num, dtype=int).tolist()
        objpoints = np.array(objpoints)[sample_idx]
        imgpoints = np.array(imgpoints)[sample_idx]
        
        print("Calibrating camera intrinsic...")
        _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (self.frame_width, self.frame_height), None, None)
        self.K = mtx
        self.D = dist

        return mtx, dist
    
    def undistort_image(self, image):
        height, width = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(self.K, self.D, (width, height), 1, (width, height))
        dst = cv2.undistort(image, self.K, self.D, None, newcameramtx)
        x, y, w, h = roi
        dst = dst[y: y + h, x: x + w]
        return dst

class CameraGroup:
    def __init__(self, 
                 cap_ids = [0, 1], 
                 resolutions = [(1280, 720), (1280, 720)], 
                 camera_group_info_path = None):
        
        self.cameras = []
        if camera_group_info_path == None:
            self.camera_num = len(cap_ids)
            for cap_id, resolution in zip(cap_ids, resolutions):
                self.cameras.append(Camera(cap_id=cap_id, frame_width=resolution[0], frame_height=resolution[1]))
        else:
            with open(camera_group_info_path, "r") as readfile:
                camera_group_info_dict = json.load(readfile)
                self.camera_num = camera_group_info_dict["camera_num"]
                for camera_info_dict in camera_group_info_dict["camera_group_info"]:
                    self.cameras.append(Camera(camera_info_dict=camera_info_dict))

    def camera_group_info_dict(self):
        camera_group_info = []
        for i in range(self.camera_num):
            camera_group_info.append(self.cameras[i].camera_info_dict())
        camera_group_info_dict = {"camera_num" : self.camera_num,
                                  "camera_group_info" : camera_group_info}
        return camera_group_info_dict

    def save_camera_group_info(self, camera_group_info_path):
        camera_group_info_dict = self.camera_group_info_dict()
        with open(camera_group_info_path, "w") as outfile:
            outfile.write(json.dumps(camera_group_info_dict, indent=4))

    def capture_init(self):
        for camera in self.cameras:
            camera.capture_init()

    def snapshot(self):
        frames = []
        for camera in self.cameras:
            frame = camera.snapshot()
            frames.append(frame)
        return frames
    
    def record_init(self, video_name, video_format = ".avi", fps = 30.0):
        for camera in self.cameras:
            camera.record_init(video_name + str(camera.cap_id), video_format, fps)

    def recording(self, images):
        for camera, image in zip(self.cameras, images):
            camera.recording(image)

    def record_end(self):
        for camera in self.cameras:
            camera.record_end()

    def intrinsic_calibrate_video(self, caps, lengths, chessboard : ChessBoard, sample_num = 40, show_vid = True):
        print("Start calibrating total " + str(len(caps)) + " cameras")
        for c, cap in enumerate(caps):
            objpoints = []
            imgpoints = [] 
            for i in range(lengths[c]):
                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, (chessboard.height, chessboard.width), None)
                if show_vid:
                    if ret:
                        cv2.drawChessboardCorners(frame, (chessboard.height, chessboard.width), corners, ret)
                    cv2.imshow("Intrinsic Calibration", frame)
                    cv2.waitKey(1)
                if ret == False:
                    continue  
                print("Get chessboard at frame " + str(i))
                corners_subpix = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), chessboard.criteria)
                objpoints.append(chessboard.objp)
                imgpoints.append(corners_subpix)

            sample_idx = np.arange(0, len(imgpoints), len(imgpoints) // sample_num, dtype=int).tolist()
            objpoints = np.array(objpoints)[sample_idx]
            imgpoints = np.array(imgpoints)[sample_idx]
        
            print("Calibrating camera intrinsic...")
            _, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, (self.cameras[c].frame_width, self.cameras[c].frame_height), None, None)
            self.cameras[c].K = mtx
            self.cameras[c].D = dist

            cv2.destroyAllWindows()

    def add_2D_points(self, points, camera_index, ax = None):
        R = self.cameras[camera_index].R
        K = self.cameras[camera_index].K
        t = self.cameras[camera_index].t
        for p in points:
            w = np.array([p[0], p[1], 1])
            f = np.dot(np.linalg.inv(K), w)
            f = np.dot(R, f)
            if ax != None:
                a = f * 10
                ax.quiver(t[0], t[1], t[2], a[0], a[1], a[2])
            self.Camera[camera_index].points.append(p)
            self.Camera[camera_index].point_rays.append(f.reshape((-1, 1)))

    def add_human_2D_points(self, person, scores, camera_index, ax = None):
        R = self.cameras[camera_index].R
        K = self.cameras[camera_index].K
        t = self.cameras[camera_index].t
        person_ray_temp = []
        person_ray_score_temp = []
        for p, s in zip(person, scores):
            w = np.array([p[0], p[1], 1])
            f = np.dot(np.linalg.inv(K), w)
            f = np.dot(R, f)
            f = f.reshape((-1, 1))
            if ax != None:
                a = f * 10
                ax.quiver(t[0], t[1], t[2], a[0], a[1], a[2])
            person_ray_temp.append(f)
            person_ray_score_temp.append(s)

        self.cameras[camera_index].hrnet_points.append(person)
        self.cameras[camera_index].hrnet_point_rays.append(person_ray_temp)
        self.cameras[camera_index].hrnet_point_score.append(person_ray_score_temp)
    
    def clear_2D_points(self):
        for camera in self.cameras:
            camera.points = []
            camera.point_rays = []
            camera.hrnet_points = []
            camera.hrnet_point_rays = []
            camera.hrnet_point_score = []

    def Human_Bundle_Adjustment(self, W_list, w_list, camera_indices, point_indices):
        J = bundle_adjustment_sparsity(self.camera_num, len(W_list), camera_indices, point_indices)
        x0 = self.BA_x0(W_list)
        e0 = self.BA_Reprojecion_Error(x0, W_list, w_list, camera_indices, point_indices)
        res = least_squares(self.BA_Reprojecion_Error, x0, jac_sparsity=J, verbose=2, ftol=1e-8, method='trf',
        args=(W_list, w_list, camera_indices, point_indices))
        e = self.BA_Reprojecion_Error(res['x'], W_list, w_list, camera_indices, point_indices)

        print("Before")
        print(np.sum(np.abs(e0)))
        print("After")
        print(np.sum(np.abs(e)))

    def BA_Reprojecion_Error(self, params, W_list, w_list, c_list, p_list):
        param_len = 7
        r_index = 4
        cam_params = params[:self.camera_num * param_len].reshape((self.camera_num, param_len))
        cam_params = cam_params[1:]
        for i, param in enumerate(cam_params, start=1):
            q = param[:r_index]
            t = param[r_index:].reshape((-1, 1))
            self.cameras[i].R = Quaternion_to_Rotation_Matrix(q)
            self.cameras[i].t = t

        e = []
        for p, w, c in zip(p_list, w_list, c_list):
            Km = self.cameras[c].K
            Rm = self.cameras[c].R
            tm = self.cameras[c].t
            em = Reprojection_Error(W_list[p], Km, Rm, tm, w)
            e.append(em[0][0])
            e.append(em[1][0])         
        return e

    def BA_x0(self, W_list):
        x0 = np.array([])
        for i in range(self.camera_num):
            R = self.cameras[i].R
            q = Rotation_Matrix_to_Quaternion(R)
            t = self.cameras[i].t
            x0 = np.append(x0, q)
            x0 = np.append(x0, t)
        x0 = np.hstack((x0, np.array(W_list).ravel()))
        return x0
    
def Quaternion_to_Rotation_Matrix(q):
    r = Rotation.from_quat(q)
    return r.as_matrix()

def Rotation_Matrix_to_Quaternion(R):
    r = Rotation.from_matrix(R)
    return r.as_quat()

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 7 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(7):
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 7 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1
    return A

def Reprojection_Error(W, K, R, t, w):
    W = W.reshape((-1, 1))
    R_inv = np.linalg.inv(R)
    W_T = np.matmul(R_inv, W - t)
    sw = np.matmul(K, W_T)
    s = sw[2]
    w_re = sw / s
    w_re = w_re.reshape(3)[:2]
    e = w - w_re
    # if np.any(np.isnan(e)) or np.any(np.isinf(e)):
    #     e = np.zeros(2)
    return e.reshape((-1, 1))