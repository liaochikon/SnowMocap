import os
import sys
import time
import numpy as np
import torch
import cv2
import snowvision as sv
import matplotlib.pyplot as plt
import socket, pickle

torch.backends.cudnn.deterministic = True
modelpath = os.path.join(os.getcwd(), "HRNet")
sys.path.insert(1, modelpath)
sys.path.insert(2, os.path.join(os.getcwd(), "HRNet/models/detectors/yolo"))
yolo_model_def=os.path.join(modelpath, "models/detectors/yolo/config/yolov3.cfg")
yolo_class_path=os.path.join(modelpath, "models/detectors/yolo/data/coco.names")
yolo_weights_path=os.path.join(modelpath, "models/detectors/yolo/weights/yolov3.weights")
HRnet_weight_path=os.path.join(modelpath, "./weights/pose_hrnet_w48_384x288.pth")

from SimpleHRNet import SimpleHRNet
from misc.visualization import draw_points_and_skeleton, joints_dict

model = SimpleHRNet(
    48,
    17,
    HRnet_weight_path,
    model_name='HRNet',
    resolution=(384, 288),
    multiperson=True,
    return_bounding_boxes=True,
    max_batch_size=16,
    yolo_model_def=yolo_model_def,
    yolo_class_path=yolo_class_path,
    yolo_weights_path=yolo_weights_path,
    device=torch.device('cuda')
)

Workspace = [-4, 4, -4, 4, 0, 8]

CameraGroup = sv.CameraGroup("snow_output_BA_Floor.txt")

ip="192.168.50.237"
Ports = [6666 , 6667, 6668, 6669]
Socs = []
for port in Ports:
    soc=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
    soc.bind((ip, port))
    Socs.append(soc)

blender_port = 6680
blender_soc=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
blender_soc.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)

img_shape = (720, 1280)

while True:
    a = time.time()
    img0 = cv2.imdecode(pickle.loads(Socs[0].recvfrom(1000000)[0]), cv2.IMREAD_COLOR)
    img1 = cv2.imdecode(pickle.loads(Socs[1].recvfrom(1000000)[0]), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(pickle.loads(Socs[2].recvfrom(1000000)[0]), cv2.IMREAD_COLOR)
    img3 = cv2.imdecode(pickle.loads(Socs[3].recvfrom(1000000)[0]), cv2.IMREAD_COLOR)

    img = np.vstack((img0, img1))
    img = np.vstack((img, img2))
    img = np.vstack((img, img3))

    _, results = model.predict(img)

    result_window = [[], [], [], []]
    for pid, person in enumerate(results):
        img = draw_points_and_skeleton(img, person, joints_dict()["coco"]['skeleton'], person_index=pid,
                                         points_color_palette='gist_rainbow', skeleton_color_palette='jet',
                                         points_palette_samples=10)
        scores = person[:, 2].reshape((-1, 1))

        person = np.flip(person[:, :2], axis=1)
        neck_point = (person[5] + person[6]) / 2
        hip_point = (person[11] + person[12]) / 2
        center_point = (neck_point + hip_point) / 2

        neck_point_scores = (scores[5] + scores[6]) / 2
        hip_point_scores = (scores[11] + scores[12]) / 2
        center_point_scores = (neck_point_scores + hip_point_scores) / 2

        person = np.vstack((person, neck_point))
        person = np.vstack((person, hip_point))
        person = np.vstack((person, center_point))

        scores = np.append(scores, [neck_point_scores, hip_point_scores, center_point_scores])

        window = int(center_point[1] // img_shape[0])
        person[:, 1] -= np.full((len(person)), img_shape[0] * window)
        CameraGroup.Add_HRNet_Person_Ray(person, scores, window)

    CameraGroup.HRNet_Triangulation(0.05)
    CameraGroup.HRNet_Triangulation_Condense(0.2, condense_person_count_tol=2)
    CameraGroup.HRNet_Triangulation_Sort()
    CameraGroup.HRNet_Triangulation_ik()
    CameraGroup.Clear_Points_Ray()

    skeleton_data = pickle.dumps(CameraGroup.HRNet_Triangulate_Point)
    blender_soc.sendto((skeleton_data), (ip, blender_port))
    print(1 / (time.time() - a))