import numpy as np
import json
import copy
from snowvision.util import Rotation_Matrix_to_Quaternion
from snowvision.triangulation import SecondOrderDynamic

def save_blender_result(blender_result, file_path):
    with open(file_path, "w") as outfile:
        outfile.write(json.dumps(blender_result, indent=4))

def Get_Root_Position(point_11, point_12):
    root_position = (point_11 + point_12) / 2
    return root_position

def Get_Root_Rotation(point_5, point_6, point_11, point_12):
    pelvis_vec = point_11 - point_12
    pelvis_mid_point = (point_11 + point_12) / 2
    shoulder_mid_point = (point_5 + point_6) / 2

    x_vec_nrom = pelvis_vec / np.linalg.norm(pelvis_vec)
    y_vec_nrom = (shoulder_mid_point - pelvis_mid_point) / np.linalg.norm(shoulder_mid_point - pelvis_mid_point)
    z_vec_nrom = np.cross(x_vec_nrom, y_vec_nrom) / np.linalg.norm(np.cross(x_vec_nrom, y_vec_nrom))

    R = np.array([x_vec_nrom,
                  y_vec_nrom,
                  z_vec_nrom]).T
    
    quat = Rotation_Matrix_to_Quaternion(R)
    quat = np.array([quat[3], quat[0], quat[1], quat[2]])

    #ret = True
    #if True in np.isnan(quat):
    #    ret = False
    #    #quat = np.array([1, 0, 0, 0])

    return quat

def Get_Chest_IK(point_5, point_6):
    chest_ik = (point_5 + point_6) / 2
    return chest_ik

def Get_Chest_Pole(point_5, point_6, point_11, point_12):
    shoulder_vec = point_5 - point_6
    pelvis_mid_point = (point_11 + point_12) / 2
    shoulder_mid_point = (point_5 + point_6) / 2
    spine_vec = shoulder_mid_point - pelvis_mid_point
    chest_pole = shoulder_mid_point + np.cross(shoulder_vec, spine_vec) / np.linalg.norm(np.cross(shoulder_vec, spine_vec))
    return chest_pole

def Get_Head_IK(point_3, point_4, point_5, point_6):
    ear_mid_point = (point_3 + point_4) / 2
    shoulder_mid_point = (point_5 + point_6) / 2
    neck_vec = ear_mid_point - shoulder_mid_point
    head_ik = shoulder_mid_point + (neck_vec) / np.linalg.norm(neck_vec)
    return head_ik

def Get_Head_Pole(point_3, point_4, point_5, point_6):
    ear_vec = point_3 - point_4
    ear_mid_point = (point_3 + point_4) / 2
    shoulder_mid_point = (point_5 + point_6) / 2
    neck_vec = ear_mid_point - shoulder_mid_point
    head_pole = ear_mid_point + np.cross(ear_vec, neck_vec) / np.linalg.norm(np.cross(ear_vec, neck_vec))
    return head_pole

def Get_Hand_Pole(index_root_point, pinky_root_point, palm_root_point, is_left):
    index_root_vec = index_root_point - palm_root_point
    pinky_root_vec = pinky_root_point - palm_root_point
    if is_left:
        hand_pole = palm_root_point + np.cross(pinky_root_vec, index_root_vec) / np.linalg.norm(np.cross(pinky_root_vec, index_root_vec))
        return hand_pole
    else:
        hand_pole = palm_root_point + np.cross(index_root_vec, pinky_root_vec) / np.linalg.norm(np.cross(index_root_vec, pinky_root_vec))
        return hand_pole
def Get_Foot_IK(thumb_point, pinky_point):
    foot_ik = (thumb_point + pinky_point) / 2
    return foot_ik

def Get_Foot_Pole(thumb_point, pinky_point, root_point, is_left):
    thumb_vec = thumb_point - root_point
    pinky_vec = pinky_point - root_point
    if is_left:
        foot_pole = root_point + np.cross(pinky_vec, thumb_vec) / np.linalg.norm(np.cross(pinky_vec, thumb_vec))
        return foot_pole
    else:
        foot_pole = root_point + np.cross(thumb_vec, pinky_vec) / np.linalg.norm(np.cross(thumb_vec, pinky_vec))
        return foot_pole
    
def Get_Joint_Pole(joint_point, upper_point, lower_point):
    a = upper_point - joint_point
    b = lower_point - joint_point
    c = upper_point - lower_point
    d = np.cross(b, a)
    n_vec = np.cross(d, c)
    n_vec_nrom = n_vec / np.linalg.norm(n_vec)
    joint_pole = joint_point + n_vec_nrom
    return joint_pole

def Human_Triangulation_Blender(result, blender_armature_profile):
    blender_result = {'blender_armature_control_points' : [], 'blender_armature_control_points_scores' : []}
    
    for person, score in zip(result['hrnet_triangulate_points'], result['hrnet_triangulate_keypoint_scores']):
        control_point_list = copy.deepcopy(blender_armature_profile)
        control_point_score_list = copy.deepcopy(blender_armature_profile)

        hand_r_ik     = person[121]
        hand_r_pole   = Get_Hand_Pole(person[117], person[129], person[112], False)
        hand_l_ik     = person[100]
        hand_l_pole   = Get_Hand_Pole(person[96], person[108], person[91], True)
        foot_r_ik     = Get_Foot_IK(person[20], person[21])
        foot_r_pole   = Get_Foot_Pole(person[20], person[21], person[22], False)
        foot_l_ik     = Get_Foot_IK(person[17], person[18])
        foot_l_pole   = Get_Foot_Pole(person[17], person[18], person[19], True)
        
        arm_r_ik      = person[10]
        arm_r_pole    = Get_Joint_Pole(person[8], person[6], person[10])
        arm_l_ik      = person[9]
        arm_l_pole    = Get_Joint_Pole(person[7], person[5], person[9])
        leg_r_ik      = person[16]
        leg_r_pole    = Get_Joint_Pole(person[14], person[12], person[16])
        leg_l_ik      = person[15]
        leg_l_pole    = Get_Joint_Pole(person[13], person[11], person[15])
        clavicle_r_ik = person[6]
        clavicle_l_ik = person[5]

        root_position = Get_Root_Position(person[11], person[12])
        root_rotation = Get_Root_Rotation(person[5], person[6], person[11], person[12])
        chest_ik      = Get_Chest_IK(person[5], person[6])
        chest_pole    = Get_Chest_Pole(person[5], person[6], person[11], person[12])
        head_ik       = Get_Head_IK(person[3], person[4], person[5], person[6])
        head_pole     = Get_Head_Pole(person[3], person[4], person[5], person[6])
    
        for control_point_name in blender_armature_profile.keys():
            control_bone = eval(control_point_name)
            control_point_list[control_point_name] = control_bone.tolist()
            if True in np.isnan(control_bone):
                control_point_score_list[control_point_name] = 0
            else:
                control_point_score_list[control_point_name] = 1
        
        blender_result['blender_armature_control_points'].append(control_point_list)
        blender_result['blender_armature_control_points_scores'].append(control_point_score_list)

    return blender_result

def Human_Triangulation_Blender_Smooth(current_blender_result, blender_armature_profile, blender_smooth_profile, previous_blender_result = None, delta_time = 1 / 30):
    blender_result = {'blender_armature_control_points' : [],
                      'blender_armature_control_points_scores' : [],
                      'second_order_dynamics' : []}
    
    if isinstance(previous_blender_result, dict):
        for control_point_list, second_order_dynamic_list, score_list in zip(current_blender_result['blender_armature_control_points'], 
                                                                             previous_blender_result['second_order_dynamics'], 
                                                                             current_blender_result['blender_armature_control_points_scores']):
            soded_control_point_list = copy.deepcopy(blender_armature_profile)
            for control_point_name in blender_armature_profile.keys():
                sod = second_order_dynamic_list[control_point_name]
                if score_list[control_point_name]:
                    soded_control_point_list[control_point_name] = sod.update(delta_time, np.array(control_point_list[control_point_name])).tolist()
                else:
                    soded_control_point_list[control_point_name] = sod.update(delta_time, sod.xp).tolist()
            blender_result['blender_armature_control_points'].append(soded_control_point_list)
        blender_result['second_order_dynamics'] = previous_blender_result['second_order_dynamics']
        blender_result['blender_armature_control_points_scores'] =   current_blender_result['blender_armature_control_points_scores']
    else:
        for control_point_list, score_list in zip(current_blender_result['blender_armature_control_points'], 
                                                  current_blender_result['blender_armature_control_points_scores']):
            sod_list = copy.deepcopy(blender_armature_profile)
            for control_point_name in blender_armature_profile.keys():
                f, z, r = blender_smooth_profile[control_point_name]
                if score_list[control_point_name]:
                    sod_list[control_point_name] = SecondOrderDynamic(f, z, r, np.array(control_point_list[control_point_name]))
                else:
                    sod_list[control_point_name] = SecondOrderDynamic(f, z, r, np.zeros(len(control_point_list[control_point_name])))
            blender_result['second_order_dynamics'].append(sod_list)
        blender_result['blender_armature_control_points'] = current_blender_result['blender_armature_control_points']
        blender_result['blender_armature_control_points_scores'] = current_blender_result['blender_armature_control_points_scores']

    return blender_result

def Human_Triangulation_To_Blender_Result(result):
    blender_result = {'armature' : [], 'score' : []}
    
    for control_points, control_points_scores in zip(result['blender_armature_control_points'], result['blender_armature_control_points_scores']):
        blender_result['armature'].append(control_points)
        blender_result['score'].append(control_points_scores)

    return blender_result