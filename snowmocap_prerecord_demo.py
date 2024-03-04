import os
import pickle
import socket
import sys
import numpy as np
import torch
import cv2
import snowvision as sv
import matplotlib.pyplot as plt

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

ip="192.168.50.237"
blender_port = 6680
blender_soc=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
blender_soc.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)

Workspace = [-2.5, 2.5, -2.5, 2.5, 0, 3]

fig = plt.figure()
ax = fig.gca(projection='3d')
CameraGroup = sv.CameraGroup("snow_output_BA_Floor.txt")

img_shape = (720, 1280)
        
cap_array, length_array, img_size_array = sv.Load_Video("video/katana1", 4)
loss = []
for i in range(length_array[0]):
    plt.cla()
    CameraGroup.Draw_CameraGroup(ax, f=0.5)
    ret, img0 = cap_array[0].read()
    ret, img1 = cap_array[1].read()
    ret, img2 = cap_array[2].read()
    ret, img3 = cap_array[3].read()
    #img0 = cv2.undistort(img0, CameraGroup.Camera[0].K, CameraGroup.Camera[0].D)
    #img1 = cv2.undistort(img1, CameraGroup.Camera[1].K, CameraGroup.Camera[1].D)
    #img2 = cv2.undistort(img2, CameraGroup.Camera[2].K, CameraGroup.Camera[2].D)
    #img3 = cv2.undistort(img3, CameraGroup.Camera[3].K, CameraGroup.Camera[3].D)
    img = np.vstack((img0, img1))
    img = np.vstack((img, img2))
    img = np.vstack((img, img3))
    _, results = model.predict(img)
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

    CameraGroup.Draw_Skeleton(ax, 0)
    skeleton_data = pickle.dumps(CameraGroup.HRNet_Triangulate_Point)
    blender_soc.sendto((skeleton_data), (ip, blender_port))
    CameraGroup.Clear_Points_Ray()
    
    ax.set_xlim3d(Workspace[0], Workspace[1])
    ax.set_ylim3d(Workspace[2], Workspace[3])
    ax.set_zlim3d(Workspace[4], Workspace[5])
    ax.set_xlabel('x')
    ax.set_ylabel('y') 
    plt.pause(0.001)
    if cv2.waitKey(1) == ord('q'):
        break
    plt.show(block=False)
    cv2.imshow(str(0), img)

blender_soc.close()
cap_array[0].release()
cap_array[1].release()
cap_array[2].release()
cap_array[3].release()