from snowvision import CameraGroup, ChessBoard, Load_Config_Json, Load_Video
from argparse import ArgumentParser

chessboard = ChessBoard()

parser = ArgumentParser()
parser.add_argument("config_path", help=": Intrinsic calibration config file")
args = parser.parse_args()

# Config loading
config = Load_Config_Json(args.config_path)
print("\nConfig loaded\n " + args.config_path)

video_dict_list = [Load_Video(video_name) for video_name in config['video_names']]
video_cap_list = [video_dict['cap'] for video_dict in video_dict_list]
video_length_list = [video_dict['length'] for video_dict in video_dict_list]

cameragroup = CameraGroup(cap_ids=config['cap_ids'], resolutions=config['resolutions'])
cameragroup.intrinsic_calibrate_video(video_cap_list, video_length_list, chessboard)
cameragroup.save_camera_group_info(config['camera_group_parameter_names'])