import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from rtmlib import Wholebody, BodyWithFeet, draw_skeleton
from argparse import ArgumentParser
from tqdm import tqdm
from snowvision import *
import cv2

parser = ArgumentParser()
parser.add_argument("config_path", help=": SnowMocap's fundamental config file")
args = parser.parse_args()

# Config loading
config = Load_Config_Json(args.config_path)
print("\nConfig loaded\n " + args.config_path)

# rtmlib model init
detector = Wholebody(to_openpose=False, mode='performance', backend=config['rtmlib_model_backend'], device=config['rtmlib_model_device'])
print("\nUsing rtmlib model " + config['rtmlib_model_alias'])

# CameraGroup object init
cameragroup = CameraGroup(camera_group_info_path=config['camera_group_info_path'])
print("\nCameraGroup object initialized, {} cameras in total".format(cameragroup.camera_num))

# Video loading
video_dict_list = [Load_Video(video_name) for video_name in config['video_names']]
video_length_list = [video_dict['length'] for video_dict in video_dict_list]
min_video_length = min(video_length_list)
[print("\nVideo loaded : " + video_name) for video_name in config['video_names']]

# Blender config loading
blender_smooth_profile = Load_Config_Json(config['blender_smooth_profile_path'])
blender_armature_profile = Load_Config_Json(config['blender_armature_profile_path'])
_, mocap_data_output_path = Check_If_File_Exist(config['mocap_data_output_path'])
print("\nBlender smooth profile loaded : " + config['blender_smooth_profile_path'])
print("\nBlender armature profile loaded : " + config['blender_armature_profile_path'])
print("\nMocap data output will be saved in " + mocap_data_output_path)

# Matplot figure init
fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))

previous_triangulation_result = None
previous_blender_result = None
blender_result_list = []

for i in tqdm(range(min_video_length), desc="Mocaping videos..."):
    frames = []
    batch_results = []
    for camera_index, (camera, video_dict) in enumerate(zip(cameragroup.cameras, video_dict_list)):
        ret, frame = video_dict['cap'].read()
        frame = cv2.undistort(frame, camera.K, camera.D)
        keypoints, scores = detector(frame)
        for person, score in zip(keypoints, scores):
            cameragroup.add_human_2D_points(person, score, camera_index)

        frame = draw_skeleton(frame, keypoints, scores, kpt_thr=config['keypoint_score_threshold'])

        frames.append(frame)
        batch_results.append((keypoints, scores))

    triangulation_result = Human_Triangulation(cameragroup, 
                                               keypoint_score_threshold=config['keypoint_score_threshold'],
                                               average_score_threshold=config['average_score_threshold'],
                                               distance_threshold=config['distance_threshold'])
    triangulation_result = Human_Triangulation_Condense(triangulation_result, 
                                                        condense_distance_tol=config['condense_distance_tol'], 
                                                        condense_person_num_tol=config['condense_person_num_tol'],
                                                        condense_score_tol=config['condense_score_tol'],
                                                        center_point_index=config['center_point_index'],
                                                        keypoint_num=config['keypoint_num'])
    triangulation_result = Human_Triangulation_Smooth(triangulation_result,
                                                      previous_triangulation_result,
                                                      f = config['smooth_f'], 
                                                      z = config['smooth_z'], 
                                                      r = config['smooth_r'], 
                                                      delta_time=config['smooth_delta_time'])
    previous_triangulation_result = triangulation_result
    
    blender_result = Human_Triangulation_Blender(triangulation_result, blender_armature_profile)
    blender_result = Human_Triangulation_Blender_Smooth(blender_result, 
                                                        blender_armature_profile, 
                                                        blender_smooth_profile,
                                                        previous_blender_result,
                                                        delta_time=config['smooth_delta_time'])
    previous_blender_result = blender_result
    blender_result_list.append(Human_Triangulation_To_Blender_Result(blender_result))

    if config['show_video']:
        frame_upper = cv2.hconcat([frames[0], frames[1]])
        frame_lower = cv2.hconcat([frames[2], frames[3]])
        cv2.imshow("Total", cv2.resize(cv2.vconcat([frame_upper, frame_lower]), (1280, 720)))
        cv2.waitKey(1)
    if config['show_plot']:
        plt.cla()
        Draw_Camera_Group(ax, cameragroup, f=0.2)
        Draw_Skeleton(triangulation_result, ax, 1)
        ax.set_xlim(-3, 3)
        ax.set_ylim(-3, 3)
        ax.set_zlim( 0, 6)
        plt.show(block=False)
        plt.pause(0.001)

    save_blender_result(blender_result_list, mocap_data_output_path)

    cameragroup.clear_2D_points()