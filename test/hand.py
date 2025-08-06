# import cv2
# from rtmlib import Hand, draw_skeleton

# device = 'cuda'
# backend = 'onnxruntime'  # opencv, onnxruntime
# cap = cv2.VideoCapture(0)
# openpose_skeleton = False  # True for openpose-style, False for mmpose-style
# hand = Hand(to_openpose=openpose_skeleton,
#             backend=backend,
#             device=device)

# while cap.isOpened():
#     success, frame = cap.read()

#     if not success:
#         break
#     keypoints, scores = hand(frame)
#     frame = draw_skeleton(frame,
#                              keypoints,
#                              scores,
#                              openpose_skeleton=openpose_skeleton,
#                              kpt_thr=0.43)
#     print(len(keypoints))
#     cv2.imshow('img', frame)
#     cv2.waitKey(10)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rtmlib import Wholebody, draw_skeleton, Hand
from tqdm import tqdm
from snowvision import *
import cv2

detector = Hand(to_openpose=False, backend='onnxruntime', device='cuda')
cameragroup = CameraGroup(camera_group_info_path="camera_group_intrinsic.json")
cameragroup.cameras[0].R = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0]])
cameragroup.cameras[0].t = np.array([0, 0, 0]).reshape((-1, 1))
cameragroup.cameras[1].R = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0]])
cameragroup.cameras[1].t = np.array([0, 2, 0]).reshape((-1, 1))
cameragroup.cameras[2].R = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0]])
cameragroup.cameras[2].t = np.array([0, 4, 0]).reshape((-1, 1))

video_names = [
    "ex\camera_i0_r1280x720_d20250621_175946.mp4",
    "ex\camera_i1_r1280x720_d20250621_175946.mp4",
    "ex\camera_i2_r1280x720_d20250621_175946.mp4",]


fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))


epochs = 5
for epoch in range(epochs):
    c_list = []
    w_list = []
    W_list = []
    p_list = []
    index = 0

    video_dict_list = [Load_Video(video_name) for video_name in video_names]
    video_length_list = [video_dict['length'] for video_dict in video_dict_list]
    min_video_length = min(video_length_list)


    for i in tqdm(range(min_video_length), desc="Mocaping videos..."):
        for camera_index, (camera, video_dict) in enumerate(zip(cameragroup.cameras, video_dict_list)):
            ret, frame = video_dict['cap'].read()
            frame = cv2.undistort(frame, camera.K, camera.D)
            keypoints, scores = detector(frame)
            for person, score in zip(keypoints, scores):
                cameragroup.add_human_2D_points(person, score, camera_index)
            frame = draw_skeleton(frame, keypoints, scores, kpt_thr=0.2)

        triangulation_result = Human_Triangulation(cameragroup, 
                                                keypoint_score_threshold=0.0,
                                                average_score_threshold=0,
                                                distance_threshold=float("inf"))

        triangulation_result = Human_Triangulation_Condense(triangulation_result, 
                                                            condense_distance_tol=float("inf"), 
                                                            condense_person_num_tol=0,
                                                            condense_score_tol=0,
                                                            center_point_index=0,
                                                            keypoint_num=21)
        
        if len(triangulation_result['hrnet_triangulate_points']) > 0:
            for p in triangulation_result['hrnet_triangulate_points']:
                # p(20x3)
                for pp in p:
                    # pp(3)
                    W_list.append(pp)
                    
            for p, c in zip(triangulation_result['hrnet_2d_points'], triangulation_result['hrnet_camera_indice']):
                # p(nx2x20x2), c(nx2)
                c = np.array(c)
                c = c.ravel()
                p = np.array(p)
                p = p.reshape((p.shape[0] * p.shape[1], 21, 2))
                for part in range(21):
                    for c_ref in range(cameragroup.camera_num):
                        for pp, cc in zip(p, c):
                            # pp(20x2), cc(scalar)
                            if cc == c_ref:
                                w_list.append(pp[part])
                                c_list.append(c_ref)
                                p_list.append(index)
                                break
                    index += 1
        
        plt.cla()
        Draw_Camera_Group(ax, cameragroup, f=0.2)
        # Draw_Skeleton(triangulation_result, ax, 1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim( 0, 6)
        plt.show(block=False)
        plt.pause(0.001)
        # print(triangulation_result["hrnet_triangulate_points"])
        cameragroup.clear_2D_points()

    print(np.array(W_list).shape)
    print(np.array(w_list).shape)
    print(np.array(c_list).shape)
    print(W_list)
    print(w_list)
    print(c_list)
    cameragroup.Human_Bundle_Adjustment(np.array(W_list), np.array(w_list), np.array(c_list), np.array(p_list))


    

