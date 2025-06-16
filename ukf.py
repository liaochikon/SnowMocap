import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import cholesky
import warnings
warnings.filterwarnings('ignore')

class HumanPoseUKF:
    def __init__(self, joints, connections, dt=1/30, smoothing_level='medium'):
        """
        人體姿態UKF濾波器
        
        Args:
            joints: 關節名稱列表
            connections: 關節連接關係 [(parent, child), ...]
            dt: 時間間隔
            smoothing_level: 'light', 'medium', 'heavy' 或 'custom'
        """
        self.joints = joints
        self.connections = connections
        self.dt = dt
        self.n_joints = len(joints)
        
        # 狀態向量: [位置(3), 速度(3), 四元數(4), 角速度(3)] for each joint
        self.state_dim = self.n_joints * 13  # 每個關節13個狀態變數
        self.obs_dim = self.n_joints * 3     # 觀測只有位置
        
        # 初始化狀態和協方差
        self.state = np.zeros(self.state_dim)
        self.P = np.eye(self.state_dim) * 0.1
        
        # 設置平滑化等級的預設參數
        self._set_smoothing_parameters(smoothing_level)
        
        # 過程噪聲和觀測噪聲
        self.Q = self._create_process_noise()
        self.R = np.eye(self.obs_dim) * self.obs_noise_std**2
        
        # 物理約束參數
        self.bone_lengths = {}  # 骨骼長度約束
        self.joint_limits = {}  # 關節角度限制
        self.max_velocity = 2.0  # 最大速度 (m/s)
        self.max_angular_velocity = 5.0  # 最大角速度 (rad/s)
        
        # UKF參數
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0.0
        self.lambda_param = self.alpha**2 * (self.state_dim + self.kappa) - self.state_dim
        
        # Sigma點權重
        self._compute_weights()
    
    def _set_smoothing_parameters(self, level):
        """設置不同等級的平滑化參數"""
        if level == 'light':
            # 輕度平滑 - 更響應原始數據
            self.pos_process_noise = 1e-3      # 位置過程噪聲
            self.vel_process_noise = 1e-2      # 速度過程噪聲  
            self.quat_process_noise = 1e-4     # 四元數過程噪聲
            self.ang_vel_process_noise = 1e-2  # 角速度過程噪聲
            self.obs_noise_std = 0.05          # 觀測噪聲標準差
            self.constraint_weight = 0.3       # 約束權重
            
        elif level == 'medium':
            # 中度平滑 - 平衡響應性和平滑度
            self.pos_process_noise = 1e-4
            self.vel_process_noise = 1e-3
            self.quat_process_noise = 1e-5
            self.ang_vel_process_noise = 1e-3
            self.obs_noise_std = 0.02
            self.constraint_weight = 0.5
            
        elif level == 'heavy':
            # 重度平滑 - 優先平滑度
            self.pos_process_noise = 1e-6
            self.vel_process_noise = 1e-4
            self.quat_process_noise = 1e-6
            self.ang_vel_process_noise = 1e-4
            self.obs_noise_std = 0.01
            self.constraint_weight = 0.8
            
        elif level == 'custom':
            # 自定義參數 - 需要手動設置
            self.pos_process_noise = 1e-4
            self.vel_process_noise = 1e-3
            self.quat_process_noise = 1e-5
            self.ang_vel_process_noise = 1e-3
            self.obs_noise_std = 0.02
            self.constraint_weight = 0.5
    
    def set_custom_parameters(self, pos_noise=None, vel_noise=None, quat_noise=None, 
                            ang_vel_noise=None, obs_noise=None, constraint_weight=None):
        """設置自定義參數"""
        if pos_noise is not None:
            self.pos_process_noise = pos_noise
        if vel_noise is not None:
            self.vel_process_noise = vel_noise
        if quat_noise is not None:
            self.quat_process_noise = quat_noise
        if ang_vel_noise is not None:
            self.ang_vel_process_noise = ang_vel_noise
        if obs_noise is not None:
            self.obs_noise_std = obs_noise
        if constraint_weight is not None:
            self.constraint_weight = constraint_weight
        
        # 重新創建噪聲矩陣
        self.Q = self._create_process_noise()
        self.R = np.eye(self.obs_dim) * self.obs_noise_std**2
        
    def _create_process_noise(self):
        """創建過程噪聲矩陣"""
        Q = np.zeros((self.state_dim, self.state_dim))
        
        for i in range(self.n_joints):
            base_idx = i * 13
            
            # 位置噪聲
            Q[base_idx:base_idx+3, base_idx:base_idx+3] = np.eye(3) * self.pos_process_noise
            # 速度噪聲
            Q[base_idx+3:base_idx+6, base_idx+3:base_idx+6] = np.eye(3) * self.vel_process_noise
            # 四元數噪聲
            Q[base_idx+6:base_idx+10, base_idx+6:base_idx+10] = np.eye(4) * self.quat_process_noise
            # 角速度噪聲
            Q[base_idx+10:base_idx+13, base_idx+10:base_idx+13] = np.eye(3) * self.ang_vel_process_noise
            
        return Q
    
    def _compute_weights(self):
        """計算UKF權重"""
        n = self.state_dim
        self.W_m = np.zeros(2*n + 1)
        self.W_c = np.zeros(2*n + 1)
        
        self.W_m[0] = self.lambda_param / (n + self.lambda_param)
        self.W_c[0] = self.lambda_param / (n + self.lambda_param) + (1 - self.alpha**2 + self.beta)
        
        for i in range(1, 2*n + 1):
            self.W_m[i] = 1 / (2 * (n + self.lambda_param))
            self.W_c[i] = 1 / (2 * (n + self.lambda_param))
    
    def _generate_sigma_points(self, x, P):
        """生成Sigma點"""
        n = len(x)
        sigma_points = np.zeros((2*n + 1, n))
        
        try:
            sqrt = cholesky((n + self.lambda_param) * P, lower=True)
        except:
            # 如果Cholesky分解失敗，使用特徵值分解
            eigenvals, eigenvecs = np.linalg.eigh(P)
            eigenvals = np.maximum(eigenvals, 1e-12)
            sqrt = eigenvecs @ np.diag(np.sqrt(eigenvals))
            sqrt *= np.sqrt(n + self.lambda_param)
        
        sigma_points[0] = x
        
        for i in range(n):
            sigma_points[i+1] = x + sqrt[:, i]
            sigma_points[i+1+n] = x - sqrt[:, i]
            
        return sigma_points
    
    def _process_model(self, state):
        """狀態轉移函數"""
        new_state = state.copy()
        
        for i in range(self.n_joints):
            base_idx = i * 13
            
            # 提取當前關節狀態
            pos = state[base_idx:base_idx+3]
            vel = state[base_idx+3:base_idx+6]
            quat = state[base_idx+6:base_idx+10]
            ang_vel = state[base_idx+10:base_idx+13]
            
            # 正規化四元數
            quat = quat / np.linalg.norm(quat)
            
            # 位置更新
            new_pos = pos + vel * self.dt
            
            # 速度約束
            vel_magnitude = np.linalg.norm(vel)
            if vel_magnitude > self.max_velocity:
                vel = vel * (self.max_velocity / vel_magnitude)
            
            # 角速度約束
            ang_vel_magnitude = np.linalg.norm(ang_vel)
            if ang_vel_magnitude > self.max_angular_velocity:
                ang_vel = ang_vel * (self.max_angular_velocity / ang_vel_magnitude)
            
            # 四元數更新（使用角速度）
            if ang_vel_magnitude > 1e-6:
                angle = ang_vel_magnitude * self.dt
                axis = ang_vel / ang_vel_magnitude
                delta_quat = np.array([
                    np.cos(angle/2),
                    axis[0] * np.sin(angle/2),
                    axis[1] * np.sin(angle/2),
                    axis[2] * np.sin(angle/2)
                ])
                new_quat = self._quaternion_multiply(quat, delta_quat)
            else:
                new_quat = quat
            
            # 更新狀態
            new_state[base_idx:base_idx+3] = new_pos
            new_state[base_idx+3:base_idx+6] = vel
            new_state[base_idx+6:base_idx+10] = new_quat / np.linalg.norm(new_quat)
            new_state[base_idx+10:base_idx+13] = ang_vel
        
        # 應用物理約束
        new_state = self._apply_physical_constraints(new_state)
        
        return new_state
    
    def _observation_model(self, state):
        """觀測函數（只觀測位置）"""
        obs = np.zeros(self.obs_dim)
        
        for i in range(self.n_joints):
            base_idx = i * 13
            obs_idx = i * 3
            obs[obs_idx:obs_idx+3] = state[base_idx:base_idx+3]
            
        return obs
    
    def _quaternion_multiply(self, q1, q2):
        """四元數乘法"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
    
    def _apply_physical_constraints(self, state):
        """應用物理約束"""
        constrained_state = state.copy()
        
        # 骨骼長度約束（使用約束權重進行軟約束）
        for parent_name, child_name in self.connections:
            if parent_name in self.joints and child_name in self.joints:
                parent_idx = self.joints.index(parent_name)
                child_idx = self.joints.index(child_name)
                
                parent_pos = constrained_state[parent_idx*13:parent_idx*13+3]
                child_pos = constrained_state[child_idx*13:child_idx*13+3]
                
                bone_key = f"{parent_name}-{child_name}"
                if bone_key in self.bone_lengths:
                    target_length = self.bone_lengths[bone_key]
                    current_vector = child_pos - parent_pos
                    current_length = np.linalg.norm(current_vector)
                    
                    if current_length > 1e-6:
                        # 軟約束：根據約束權重調整
                        corrected_vector = current_vector * (target_length / current_length)
                        correction = corrected_vector - current_vector
                        
                        # 應用約束權重
                        constrained_state[child_idx*13:child_idx*13+3] = (
                            child_pos + self.constraint_weight * correction
                        )
        
        return constrained_state
    
    def set_bone_length(self, parent, child, length):
        """設置骨骼長度約束"""
        self.bone_lengths[f"{parent}-{child}"] = length
    
    def set_joint_limits(self, joint_name, limits):
        """設置關節角度限制"""
        self.joint_limits[joint_name] = limits
    
    def initialize_state(self, initial_positions):
        """初始化狀態"""
        for i, pos in enumerate(initial_positions):
            base_idx = i * 13
            self.state[base_idx:base_idx+3] = pos
            # 初始四元數為單位四元數
            self.state[base_idx+6] = 1.0
    
    def predict(self):
        """預測步驟"""
        # 生成Sigma點
        sigma_points = self._generate_sigma_points(self.state, self.P)
        
        # 通過過程模型傳播Sigma點
        sigma_points_pred = np.zeros_like(sigma_points)
        for i, sp in enumerate(sigma_points):
            sigma_points_pred[i] = self._process_model(sp)
        
        # 預測狀態均值
        self.state = np.sum(self.W_m[:, np.newaxis] * sigma_points_pred, axis=0)
        
        # 預測協方差
        self.P = self.Q.copy()
        for i, sp in enumerate(sigma_points_pred):
            diff = sp - self.state
            self.P += self.W_c[i] * np.outer(diff, diff)
    
    def update(self, observation):
        """更新步驟"""
        # 生成Sigma點
        sigma_points = self._generate_sigma_points(self.state, self.P)
        
        # 通過觀測模型傳播Sigma點
        sigma_obs = np.zeros((len(sigma_points), self.obs_dim))
        for i, sp in enumerate(sigma_points):
            sigma_obs[i] = self._observation_model(sp)
        
        # 預測觀測均值
        obs_pred = np.sum(self.W_m[:, np.newaxis] * sigma_obs, axis=0)
        
        # 計算創新協方差和交叉協方差
        S = self.R.copy()
        P_xy = np.zeros((self.state_dim, self.obs_dim))
        
        for i in range(len(sigma_points)):
            obs_diff = sigma_obs[i] - obs_pred
            state_diff = sigma_points[i] - self.state
            
            S += self.W_c[i] * np.outer(obs_diff, obs_diff)
            P_xy += self.W_c[i] * np.outer(state_diff, obs_diff)
        
        # 卡爾曼增益
        try:
            K = P_xy @ np.linalg.inv(S)
        except:
            K = P_xy @ np.linalg.pinv(S)
        
        # 更新狀態和協方差
        innovation = observation - obs_pred
        self.state += K @ innovation
        self.P -= K @ S @ K.T
        
        # 確保協方差矩陣正定
        eigenvals = np.linalg.eigvals(self.P)
        if np.min(eigenvals) < 1e-12:
            self.P += np.eye(self.state_dim) * 1e-12

# 使用範例
def create_sample_data():
    """創建範例人體關節數據"""
    # 定義簡化的人體關節
    joints = ['head', 'neck', 'left_shoulder', 'right_shoulder', 
              'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
              'spine', 'left_hip', 'right_hip']
    
    connections = [
        ('head', 'neck'), ('neck', 'left_shoulder'), ('neck', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
        ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
        ('neck', 'spine'), ('spine', 'left_hip'), ('spine', 'right_hip')
    ]
    
    # 生成模擬數據（帶噪聲的關節位置）
    n_frames = 100
    dt = 1/30
    
    # 初始關節位置
    initial_positions = np.array([
        [0, 0, 1.7],      # head
        [0, 0, 1.6],      # neck
        [-0.2, 0, 1.5],   # left_shoulder
        [0.2, 0, 1.5],    # right_shoulder
        [-0.4, 0, 1.3],   # left_elbow
        [0.4, 0, 1.3],    # right_elbow
        [-0.5, 0, 1.1],   # left_wrist
        [0.5, 0, 1.1],    # right_wrist
        [0, 0, 1.2],      # spine
        [-0.1, 0, 0.9],   # left_hip
        [0.1, 0, 0.9]     # right_hip
    ])
    
    # 生成運動軌跡（簡單的擺臂動作）
    trajectory = []
    for t in range(n_frames):
        frame_data = initial_positions.copy()
        
        # 添加簡單的擺臂動作
        swing_angle = np.sin(t * dt * 2 * np.pi * 0.5) * 0.3
        
        # 左臂擺動
        frame_data[4, 1] = initial_positions[4, 1] + 0.2 * np.sin(swing_angle)  # left_elbow
        frame_data[6, 1] = initial_positions[6, 1] + 0.3 * np.sin(swing_angle)  # left_wrist
        
        # 右臂擺動
        frame_data[5, 1] = initial_positions[5, 1] - 0.2 * np.sin(swing_angle)  # right_elbow
        frame_data[7, 1] = initial_positions[7, 1] - 0.3 * np.sin(swing_angle)  # right_wrist
        
        # 添加噪聲
        noise = np.random.normal(0, 0.02, frame_data.shape)
        frame_data += noise
        
        trajectory.append(frame_data)
    
    return joints, connections, trajectory

def compare_smoothing_levels():
    """比較不同平滑化等級的效果"""
    joints, connections, trajectory = create_sample_data()
    
    smoothing_levels = ['light', 'medium', 'heavy']
    filtered_trajectories = {}
    
    for level in smoothing_levels:
        print(f"處理 {level} 平滑化等級...")
        ukf = HumanPoseUKF(joints, connections, dt=1/30, smoothing_level=level)
        
        # 設置骨骼長度約束
        ukf.set_bone_length('head', 'neck', 0.1)
        ukf.set_bone_length('neck', 'left_shoulder', 0.15)
        ukf.set_bone_length('left_shoulder', 'left_elbow', 0.3)
        ukf.set_bone_length('left_elbow', 'left_wrist', 0.25)
        
        ukf.initialize_state(trajectory[0])
        
        filtered_traj = []
        for frame in trajectory:
            ukf.predict()
            ukf.update(frame.flatten())
            
            filtered_positions = np.zeros((len(joints), 3))
            for i in range(len(joints)):
                base_idx = i * 13
                filtered_positions[i] = ukf.state[base_idx:base_idx+3]
            filtered_traj.append(filtered_positions)
        
        filtered_trajectories[level] = filtered_traj
    
    # 可視化比較
    plt.figure(figsize=(20, 10))
    joint_idx = joints.index('left_wrist')
    
    colors = {'light': 'green', 'medium': 'blue', 'heavy': 'purple'}
    
    for axis, label in enumerate(['X', 'Y', 'Z']):
        plt.subplot(2, 3, axis+1)
        
        original_data = [frame[joint_idx, axis] for frame in trajectory]
        plt.plot(original_data, 'r-', alpha=0.7, linewidth=1, label='原始數據（含噪聲）')
        
        for level in smoothing_levels:
            filtered_data = [frame[joint_idx, axis] for frame in filtered_trajectories[level]]
            plt.plot(filtered_data, color=colors[level], linewidth=2, label=f'{level} 平滑')
        
        plt.title(f'左手腕 {label} 軸位置比較')
        plt.xlabel('幀數')
        plt.ylabel(f'{label} 位置 (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 計算並顯示誤差
        plt.subplot(2, 3, axis+4)
        for level in smoothing_levels:
            filtered_data = [frame[joint_idx, axis] for frame in filtered_trajectories[level]]
            error = np.array(filtered_data) - np.array(original_data)
            plt.plot(error, color=colors[level], linewidth=2, label=f'{level} 誤差')
        
        plt.title(f'{label} 軸濾波誤差')
        plt.xlabel('幀數')
        plt.ylabel('誤差 (m)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """主函數示例 - 展示自定義參數調整"""
    joints, connections, trajectory = create_sample_data()
    
    # 示例1: 使用輕度平滑
    print("=== 輕度平滑範例 ===")
    ukf_light = HumanPoseUKF(joints, connections, dt=1/30, smoothing_level='light')
    
    # 示例2: 自定義參數
    print("=== 自定義參數範例 ===")
    ukf_custom = HumanPoseUKF(joints, connections, dt=1/30, smoothing_level='custom')
    
    # 調整參數以減少平滑化
    ukf_custom.set_custom_parameters(
        pos_noise=1e-3,        # 增加位置過程噪聲 -> 更響應數據變化
        vel_noise=5e-3,        # 增加速度過程噪聲
        obs_noise=0.08,        # 增加觀測噪聲 -> 減少對測量的信任
        constraint_weight=0.2   # 降低約束權重 -> 減少約束影響
    )
    
    print(f"自定義參數:")
    print(f"  位置過程噪聲: {ukf_custom.pos_process_noise}")
    print(f"  觀測噪聲: {ukf_custom.obs_noise_std}")
    print(f"  約束權重: {ukf_custom.constraint_weight}")
    
    # 設置骨骼長度約束
    for ukf in [ukf_light, ukf_custom]:
        ukf.set_bone_length('left_shoulder', 'left_elbow', 0.3)
        ukf.set_bone_length('left_elbow', 'left_wrist', 0.25)
        ukf.initialize_state(trajectory[0])
    
    # 處理數據
    results = {}
    for name, ukf in [('light', ukf_light), ('custom', ukf_custom)]:
        filtered_traj = []
        for frame in trajectory:
            ukf.predict()
            ukf.update(frame.flatten())
            
            filtered_positions = np.zeros((len(joints), 3))
            for i in range(len(joints)):
                base_idx = i * 13
                filtered_positions[i] = ukf.state[base_idx:base_idx+3]
            filtered_traj.append(filtered_positions)
        
        results[name] = filtered_traj
    
    # 可視化對比
    plt.figure(figsize=(15, 10))
    joint_idx = joints.index('left_wrist')
    
    original_data = [frame[joint_idx] for frame in trajectory]
    
    for axis, label in enumerate(['X', 'Y', 'Z']):
        plt.subplot(2, 3, axis+1)
        plt.plot([pos[axis] for pos in original_data], 'r-', alpha=0.7, label='原始數據')
        plt.plot([frame[joint_idx, axis] for frame in results['light']], 
                'g-', linewidth=2, label='輕度平滑')
        plt.plot([frame[joint_idx, axis] for frame in results['custom']], 
                'b-', linewidth=2, label='自定義參數')
        plt.title(f'左手腕 {label} 軸位置')
        plt.xlabel('幀數')
        plt.ylabel(f'{label} 位置 (m)')
        plt.legend()
        plt.grid(True)
        
        # 響應性分析
        plt.subplot(2, 3, axis+4)
        original_vel = np.diff([pos[axis] for pos in original_data])
        light_vel = np.diff([frame[joint_idx, axis] for frame in results['light']])
        custom_vel = np.diff([frame[joint_idx, axis] for frame in results['custom']])
        
        plt.plot(original_vel, 'r-', alpha=0.7, label='原始速度')
        plt.plot(light_vel, 'g-', linewidth=2, label='輕度平滑速度')
        plt.plot(custom_vel, 'b-', linewidth=2, label='自定義參數速度')
        plt.title(f'{label} 軸速度比較')
        plt.xlabel('幀數')
        plt.ylabel(f'{label} 速度 (m/frame)')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 比較不同平滑化等級
    compare_smoothing_levels()
    
    print("\n=== 參數調整建議 ===")
    print("1. 減少平滑化:")
    print("   - 增加過程噪聲 (pos_noise, vel_noise)")
    print("   - 增加觀測噪聲 (obs_noise)")
    print("   - 降低約束權重 (constraint_weight)")
    print("\n2. 增加平滑化:")
    print("   - 降低過程噪聲")
    print("   - 降低觀測噪聲")
    print("   - 增加約束權重")
    print("\n3. 平衡調整:")
    print("   - 根據實際數據噪聲水平調整觀測噪聲")
    print("   - 根據運動類型調整過程噪聲")
    print("   - 根據約束重要性調整約束權重")

if __name__ == "__main__":
    main()