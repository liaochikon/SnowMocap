import numpy as np
import cv2
import json
from scipy.spatial.transform import Rotation
import os.path

def Check_If_File_Exist(file_path):
    is_file_exist = os.path.isfile(file_path)
    if is_file_exist:
        idx = 0
        file_path_alter = ""
        while is_file_exist:
            file_name, file_format = file_path.split(".")
            file_path_alter = file_name + "_" + str(idx) + "." + file_format
            is_file_exist = os.path.isfile(file_path_alter)
            idx += 1
        return is_file_exist, file_path_alter
    else:
        return is_file_exist, file_path

def Load_Config_Json(config_path):
    with open(config_path, "r") as readfile:
        config = json.load(readfile)
        return config

def Rotation_Matrix_to_Quaternion(R):
    r = Rotation.from_matrix(R)
    return r.as_quat()

def Quaternion_to_Rotation_Matrix(q):
    r = Rotation.from_quat(q)
    return r.as_matrix()

def Load_Video(video_name):
    cap = cv2.VideoCapture(video_name)
    length = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_COUNT))
    image_width = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_WIDTH))
    image_height = int(cv2.VideoCapture.get(cap, cv2.CAP_PROP_FRAME_HEIGHT))
    return {'cap' : cap, 
            'length' : length, 
            'image_size' : (image_width, image_height)}

def Increase_Brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img


def Draw_ChessBoard(image, board_shape):
    #board_shape = [corners[0][0], corners[8][0], corners[53][0], corners[45][0]]
    cv2.line(image,(int(board_shape[0][0]), int(board_shape[0][1])),(int(board_shape[1][0]), int(board_shape[1][1])),(0,255,255),2)
    cv2.line(image,(int(board_shape[1][0]), int(board_shape[1][1])),(int(board_shape[2][0]), int(board_shape[2][1])),(0,255,255),2)
    cv2.line(image,(int(board_shape[2][0]), int(board_shape[2][1])),(int(board_shape[3][0]), int(board_shape[3][1])),(0,255,255),2)
    cv2.line(image,(int(board_shape[3][0]), int(board_shape[3][1])),(int(board_shape[0][0]), int(board_shape[0][1])),(0,255,255),2)
    cv2.line(image,(int(board_shape[0][0]), int(board_shape[0][1])),(int(board_shape[2][0]), int(board_shape[2][1])),(0,255,255),2)
    cv2.line(image,(int(board_shape[1][0]), int(board_shape[1][1])),(int(board_shape[3][0]), int(board_shape[3][1])),(0,255,255),2)

def Draw_Camera(K, R, t, cam_num, ax, f=1):
    r0 = R[:, 0].reshape(3)
    r1 = R[:, 1].reshape(3)
    r2 = R[:, 2].reshape(3)
    ax.text(t[0][0], t[1][0], t[2][0], str(cam_num))
    ax.quiver(t[0], t[1], t[2], r0[0], r0[1], r0[2], color='r')
    ax.quiver(t[0], t[1], t[2], r1[0], r1[1], r1[2], color='g')
    ax.quiver(t[0], t[1], t[2], r2[0], r2[1], r2[2], color='b')
    vec = np.zeros(3)
    vec[0] = K[0][2] / K[0][0] * f
    vec[1] = K[1][2] / K[1][1] * f
    vec[2] = f
    t_T = t.reshape(3)
    lt = (-vec[0]) * r0 + (-vec[1]) * r1 + vec[2] * r2 + t_T
    lb = (-vec[0]) * r0 + vec[1] * r1 + vec[2] * r2 + t_T
    rt = vec[0] * r0 + (-vec[1]) * r1 + vec[2] * r2 + t_T
    rb = vec[0] * r0 + vec[1] * r1 + vec[2] * r2 + t_T
    ax.plot3D(xs=(t_T[0], lt[0]),
              ys=(t_T[1], lt[1]),
              zs=(t_T[2], lt[2]), color='k')
    ax.plot3D(xs=(t_T[0], rt[0]),
              ys=(t_T[1], rt[1]),
              zs=(t_T[2], rt[2]), color='k')
    ax.plot3D(xs=(t_T[0], lb[0]),
              ys=(t_T[1], lb[1]),
              zs=(t_T[2], lb[2]), color='k')
    ax.plot3D(xs=(t_T[0], rb[0]),
              ys=(t_T[1], rb[1]),
              zs=(t_T[2], rb[2]), color='k')

    ax.plot3D(xs=(lt[0], rt[0]),
              ys=(lt[1], rt[1]),
              zs=(lt[2], rt[2]), color='k')
    ax.plot3D(xs=(rt[0], rb[0]),
              ys=(rt[1], rb[1]),
              zs=(rt[2], rb[2]), color='k')
    ax.plot3D(xs=(rb[0], lb[0]),
              ys=(rb[1], lb[1]),
              zs=(rb[2], lb[2]), color='k')
    ax.plot3D(xs=(lb[0], lt[0]),
              ys=(lb[1], lt[1]),
              zs=(lb[2], lt[2]), color='k')

def Draw_Camera_Group(ax, camera_group, f=1):
    for i, camera in enumerate(camera_group.cameras):
        Draw_Camera(camera.K, camera.R, camera.t, i, ax, f)

def Draw_Skeleton(result, ax, max_index = 0, isbone_label = False):
    bones = {
        #face
        'ear_r':[4, 2], 
        'ear_l':[1, 3], 
        'eye_r':[2, 0], 
        'eye_l':[1, 0], 

        #body
        'body_1':[5, 6], 
        'body_2':[6, 12], 
        'body_3':[12, 11], 
        'body_4':[11, 5], 

        #arm
        'upper_arm_r':[6, 8], 
        'lower_arm_r':[8, 10], 
        'upper_arm_l':[5, 7], 
        'lower_arm_l':[7, 9], 

        #leg
        'upper_leg_r':[12, 14], 
        'lower_leg_r':[14, 16], 
        'upper_leg_l':[11, 13], 
        'lower_leg_l':[13, 15], 

        #foot
        'foot_1_r':[20, 16], 
        'foot_2_r':[21, 16], 
        'foot_3_r':[22, 16], 
        'foot_1_l':[17, 15], 
        'foot_2_l':[18, 15], 
        'foot_3_l':[19, 15], 

        #palm
        'palm_1_r':[112, 113], 
        'palm_2_r':[112, 117], 
        'palm_3_r':[112, 121], 
        'palm_4_r':[112, 125], 
        'palm_5_r':[112, 129], 
        'palm_1_l':[91, 92], 
        'palm_2_l':[91, 96], 
        'palm_3_l':[91, 100], 
        'palm_4_l':[91, 104], 
        'palm_5_l':[91, 108], 

        #finger_r
        'finger_thumb_1_r':[113, 114], 
        'finger_thumb_2_r':[114, 115], 
        'finger_thumb_3_r':[115, 116], 

        'finger_index_1_r':[117, 118], 
        'finger_index_2_r':[118, 119], 
        'finger_index_3_r':[119, 120], 

        'finger_middle_1_r':[121, 122], 
        'finger_middle_2_r':[122, 123], 
        'finger_middle_3_r':[123, 124], 

        'finger_ring_1_r':[125, 126], 
        'finger_ring_2_r':[126, 127], 
        'finger_ring_3_r':[127, 128], 

        'finger_pinky_1_r':[129, 130], 
        'finger_pinky_2_r':[130, 131], 
        'finger_pinky_3_r':[131, 132], 

        #finger_l
        'finger_thumb_1_l':[92, 93], 
        'finger_thumb_2_l':[93, 94], 
        'finger_thumb_3_l':[94, 95], 

        'finger_index_1_l':[96, 97], 
        'finger_index_2_l':[97, 98], 
        'finger_index_3_l':[98, 99], 

        'finger_middle_1_l':[100, 101], 
        'finger_middle_2_l':[101, 102], 
        'finger_middle_3_l':[102, 103], 

        'finger_ring_1_l':[104, 105], 
        'finger_ring_2_l':[105, 106], 
        'finger_ring_3_l':[106, 107], 

        'finger_pinky_1_l':[108, 109], 
        'finger_pinky_2_l':[109, 110], 
        'finger_pinky_3_l':[110, 111], 

    }
    for i, (person, scores), in enumerate(zip(result['hrnet_triangulate_points'], result['hrnet_triangulate_keypoint_scores'])):
        if i > max_index:
            break
        #ax.text(person[0][0], person[0][1], person[0][2], str(round(total_scores, 2)), size=10, zorder=1,  color='r')
        for b in bones:
            if scores[bones[b][0]] == 0 or scores[bones[b][1]] == 0:
                #print("skip")
                continue
            bone_head = np.array([person[bones[b][0]][0], person[bones[b][0]][1], person[bones[b][0]][2]])
            bone_tail = np.array([person[bones[b][1]][0], person[bones[b][1]][1], person[bones[b][1]][2]])
            
            ax.plot3D(xs=(bone_head[0], bone_tail[0]),
                      ys=(bone_head[1], bone_tail[1]),
                      zs=(bone_head[2], bone_tail[2]))
            if isbone_label:
                ax.text(bone_head[0], bone_head[1], bone_head[2], (b), size=5, zorder=1,  color='k')
        #for i, p in enumerate(person):
        #    ax.text(p[0], p[1], p[2], str(i), size=10, zorder=1,  color='b')