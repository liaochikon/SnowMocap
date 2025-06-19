from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints
from scipy.spatial.transform import Rotation as R
import numpy as np

def bfx(x, dt):
    for i in range(len(x) // 9):
        x[3 + i * 9] += x[6 + i * 9] * dt
        x[4 + i * 9] += x[7 + i * 9] * dt
        x[5 + i * 9] += x[8 + i * 9] * dt
        x[0 + i * 9] += x[3 + i * 9] * dt
        x[1 + i * 9] += x[4 + i * 9] * dt
        x[2 + i * 9] += x[5 + i * 9] * dt
    return x

def bhx(x):
    pos = []
    for i in range(len(x) // 9):
        pos.append(x[0 + i * 9])
        pos.append(x[1 + i * 9])
        pos.append(x[2 + i * 9])
    return np.array(pos)

class BUKF():
    def __init__(self):
        self.joint_num = 24
        self.points = MerweScaledSigmaPoints(n=9*self.joint_num, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UnscentedKalmanFilter(dim_x=9*self.joint_num, dim_z=3*self.joint_num, dt=0.033, fx=fx, hx=hx, points=self.points)
        self.ukf.Q = np.eye(9*self.joint_num) * 1e-2  
        self.ukf.R = np.eye(3*self.joint_num) * 1e-2
        self.ukf.R[5 * 3 + 0][5 * 3 + 0] = 3
        self.ukf.R[5 * 3 + 1][5 * 3 + 1] = 3
        self.ukf.R[5 * 3 + 2][5 * 3 + 2] = 3
        self.ukf.R[7 * 3 + 0][7 * 3 + 0] = 3
        self.ukf.R[7 * 3 + 1][7 * 3 + 1] = 3
        self.ukf.R[7 * 3 + 2][7 * 3 + 2] = 3
        self.ukf.R[9 * 3 + 0][9 * 3 + 0] = 3
        self.ukf.R[9 * 3 + 1][9 * 3 + 1] = 3
        self.ukf.R[9 * 3 + 2][9 * 3 + 2] = 3
        self.ukf.R[11 * 3 + 0][11 * 3 + 0] = 3
        self.ukf.R[11 * 3 + 1][11 * 3 + 1] = 3
        self.ukf.R[11 * 3 + 2][11 * 3 + 2] = 3
        self.ukf.x = np.zeros(9*self.joint_num)
        self.ukf.P = np.eye(9*self.joint_num) * 1e-2

    def blender_to_ukf(self, blender_data):
        frame = []
        for k in blender_data[0].keys():
            pos = blender_data[0][k]
            if any(np.isnan(pos)):
                pos = np.zeros(3)
            frame.extend(pos)
        return np.array(frame)
    
    def ukf_to_blender(self, ukf, keys):
        blender_data = {}
        for i, k in enumerate(keys):
            blender_data[k] = ukf[i * 3 : i * 3 + 3].tolist()
        return blender_data
    
    def update(self, blender_data):
        frame = self.blender_to_ukf(blender_data)
        self.ukf.predict()
        self.ukf.update(frame)
        f = bhx(self.ukf.x)
        f = self.ukf_to_blender(f, blender_data[0].keys())
        return [f]

def fx(x, dt):
    for i in range(len(x) // 9):
        x[3 + i * 9] += x[6 + i * 9] * dt
        x[4 + i * 9] += x[7 + i * 9] * dt
        x[5 + i * 9] += x[8 + i * 9] * dt
        x[0 + i * 9] += x[3 + i * 9] * dt
        x[1 + i * 9] += x[4 + i * 9] * dt
        x[2 + i * 9] += x[5 + i * 9] * dt
    return x

def hx(x):
    pos = []
    for i in range(len(x) // 9):
        pos.append(x[0 + i * 9])
        pos.append(x[1 + i * 9])
        pos.append(x[2 + i * 9])
    return np.array(pos)

class UKF():
    def __init__(self):
        self.joint_num = 28
        self.points = MerweScaledSigmaPoints(n=9*self.joint_num, alpha=0.1, beta=2.0, kappa=0.0)
        self.ukf = UnscentedKalmanFilter(dim_x=9*self.joint_num, dim_z=3*self.joint_num, dt=0.033, fx=fx, hx=hx, points=self.points)
        self.ukf.Q = np.eye(9*self.joint_num) * 1e-2 * 0.5  
        self.ukf.R = np.eye(3*self.joint_num) * 1e-2
        self.ukf.x = np.zeros(9*self.joint_num)
        self.ukf.P = np.eye(9*self.joint_num) * 1e-2

    def update(self, frame):
        z = np.zeros((28, 3))
        z[0] = frame[3]
        z[1] = frame[4]
        z[2] = frame[5]
        z[3] = frame[6]
        z[4] = frame[7]
        z[5] = frame[8]
        z[6] = frame[9]
        z[7] = frame[10]
        z[8] = frame[11]
        z[9] = frame[12]
        z[10] = frame[13]
        z[11] = frame[14]
        z[12] = frame[15]
        z[13] = frame[16]
        z[14] = frame[17]
        z[15] = frame[18]
        z[16] = frame[19]
        z[17] = frame[20]
        z[18] = frame[21]
        z[19] = frame[22]
        z[20] = frame[91]
        z[21] = frame[96]
        z[22] = frame[100]
        z[23] = frame[108]
        z[24] = frame[112]
        z[25] = frame[117]
        z[26] = frame[121]
        z[27] = frame[129]

        self.ukf.predict()
        self.ukf.update(z.flatten())
        f = hx(self.ukf.x).reshape(28, 3)
        z = np.zeros((133, 3))
        z[3]   = f[0] 
        z[4]   = f[1] 
        z[5]   = f[2] 
        z[6]   = f[3] 
        z[7]   = f[4] 
        z[8]   = f[5] 
        z[9]   = f[6] 
        z[10]  = f[7] 
        z[11]  = f[8] 
        z[12]  = f[9] 
        z[13]  = f[10]
        z[14]  = f[11]
        z[15]  = f[12]
        z[16]  = f[13]
        z[17]  = f[14]
        z[18]  = f[15]
        z[19]  = f[16]
        z[20]  = f[17]
        z[21]  = f[18]
        z[22]  = f[19]
        z[91]  = f[20]
        z[96]  = f[21]
        z[100] = f[22]
        z[108] = f[23]
        z[112] = f[24]
        z[117] = f[25]
        z[121] = f[26]
        z[129] = f[27]

        return z