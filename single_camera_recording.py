import cv2
from snowvision import Camera, Load_Config_Json
from argparse import ArgumentParser
from datetime import datetime

parser = ArgumentParser()
parser.add_argument("config_path", help=": Single camera recording config file")
args = parser.parse_args()

config = Load_Config_Json(args.config_path)
print("\nConfig loaded\n " + args.config_path)

camera_1 = Camera(config['cap_id'], frame_width=config['resolution'][0], frame_height=config['resolution'][1])
camera_1.capture_init()

now = datetime.now()
date_time = now.strftime("%m%d%Y%H%M%S")
camera_1.record_init(date_time)
while True:
    frame = camera_1.snapshot()
    camera_1.recording(frame)
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

camera_1.record_end()