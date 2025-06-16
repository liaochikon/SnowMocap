from snowvision import *
from ukf import HumanPoseUKF
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

raw_data = Load_Config_Json("test_data.json")['test']

fig = plt.figure()
ax = fig.add_axes(Axes3D(fig))

test_data = []
for _data in raw_data:
    _data = np.array(_data[0])
    pos = []
    pos.append(_data[0])#'head'
    pos.append(_data[1])#'left_eye'
    pos.append(_data[2])#'right_eye'
    pos.append(_data[3])#'left_ear'
    pos.append(_data[4])#'right_ear'
    pos.append(np.array([#'neck'
        (_data[5][0] + _data[6][0]) / 2, 
        (_data[5][1] + _data[6][1]) / 2, 
        (_data[5][2] + _data[6][2]) / 2]))
    pos.append(_data[5])#'left_shoulder'
    pos.append(_data[6])#'right_shoulder'
    pos.append(_data[7])#'left_elbow'
    pos.append(_data[8])#'right_elbow'
    pos.append(_data[9])#'left_wrist'
    pos.append(_data[10])#'right_wrist'
    pos.append(np.array([#'spine'
        (_data[11][0] + _data[12][0]) / 2, 
        (_data[11][1] + _data[12][1]) / 2, 
        (_data[11][2] + _data[12][2]) / 2]))
    pos.append(_data[11])#'left_hip'
    pos.append(_data[12])#'right_hip'
    pos.append(_data[13])#'left_knee'
    pos.append(_data[14])#'right_knee'
    pos.append(_data[15])#'left_ankle'
    pos.append(_data[16])#'right_ankle'

    test_data.append(np.array(pos))

joints = [
    'head', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'neck', 
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'spine', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

connections = [
    ('head', 'neck'), 
    ('head', 'left_eye'), ('head', 'right_eye'), 
    ('left_eye', 'left_ear'), ('right_eye', 'right_ear'), 
    ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('neck', 'spine'), ('spine', 'left_hip'), ('spine', 'right_hip'), 
    ('left_hip', 'left_knee'), ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'), ('right_knee', 'right_ankle')]

ukf = HumanPoseUKF(joints, connections, 1/30)

first_frame_data = test_data[0]
ukf.set_bone_length('head', 'neck', np.linalg.norm(first_frame_data[0] - first_frame_data[5]))
ukf.set_bone_length('head', 'left_eye', np.linalg.norm(first_frame_data[0] - first_frame_data[1]))
ukf.set_bone_length('head', 'right_eye', np.linalg.norm(first_frame_data[0] - first_frame_data[2]))
ukf.set_bone_length('left_eye', 'left_ear', np.linalg.norm(first_frame_data[1] - first_frame_data[3]))
ukf.set_bone_length('right_eye', 'right_ear', np.linalg.norm(first_frame_data[2] - first_frame_data[4]))
ukf.set_bone_length('neck', 'left_shoulder', np.linalg.norm(first_frame_data[5] - first_frame_data[6]))
ukf.set_bone_length('neck', 'right_shoulder', np.linalg.norm(first_frame_data[5] - first_frame_data[7]))
ukf.set_bone_length('left_shoulder', 'left_elbow', np.linalg.norm(first_frame_data[6] - first_frame_data[8]))
ukf.set_bone_length('left_elbow', 'left_wrist', np.linalg.norm(first_frame_data[8] - first_frame_data[10]))
ukf.set_bone_length('right_shoulder', 'right_elbow', np.linalg.norm(first_frame_data[7] - first_frame_data[9]))
ukf.set_bone_length('right_elbow', 'right_wrist', np.linalg.norm(first_frame_data[9] - first_frame_data[11]))
ukf.set_bone_length('neck', 'spine', np.linalg.norm(first_frame_data[5] - first_frame_data[12]))
ukf.set_bone_length('spine', 'left_hip', np.linalg.norm(first_frame_data[12] - first_frame_data[13]))
ukf.set_bone_length('spine', 'right_hip', np.linalg.norm(first_frame_data[12] - first_frame_data[14]))
ukf.set_bone_length('left_hip', 'left_knee', np.linalg.norm(first_frame_data[13] - first_frame_data[15]))
ukf.set_bone_length('right_hip', 'right_knee', np.linalg.norm(first_frame_data[14] - first_frame_data[16]))
ukf.set_bone_length('left_knee', 'left_ankle', np.linalg.norm(first_frame_data[15] - first_frame_data[17]))
ukf.set_bone_length('right_knee', 'right_ankle', np.linalg.norm(first_frame_data[16] - first_frame_data[18]))

ukf.initialize_state(test_data[0])

# 濾波處理
filtered_trajectory = []

for idx, frame in enumerate(test_data):

    print(idx)
    ukf.predict()
    
    observation = frame.flatten()
    ukf.update(observation)
    
    filtered_positions = np.zeros((len(joints), 3))
    for i in range(len(joints)):
        base_idx = i * 13
        filtered_positions[i] = ukf.state[base_idx:base_idx+3]
    
    filtered_trajectory.append(filtered_positions)
    

for f in filtered_trajectory:
    ax.scatter(f[:, 0], f[:, 1], f[:, 2], c='red')

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim( 0, 6)

    plt.show(block=False)
    plt.pause(0.001)
    plt.cla()