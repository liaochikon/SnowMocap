import cv2
from datetime import datetime
from snowvision import CameraGroup, Load_Config_Json
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("config_path", help=": Multi camera recording config file")
args = parser.parse_args()

config = Load_Config_Json(args.config_path)
print("\nConfig loaded\n " + args.config_path)

cameragroup = CameraGroup(cap_ids=config['cap_ids'], resolutions=config['resolutions'])
cameragroup.capture_init()

recording = False
show_index = 0
while True:
    frames = cameragroup.snapshot()
    if recording:
        cameragroup.recording(frames)
    for i, frame in enumerate(frames):
        if i == show_index:
            cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('i'):
        show_index += 1
        if show_index >= (cameragroup.camera_num):
            show_index = 0
    elif key == ord('r'):
        if recording:
            cameragroup.record_end()
            recording = False
            print("End recording")
        else:
            now = datetime.now()
            date_time = now.strftime("%m%d%Y%H%M%S")
            cameragroup.record_init(date_time)
            recording = True
            print("Start recording")
    elif key == ord('m'):
        print("change mode")