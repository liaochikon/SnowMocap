import numpy as np
from scipy.optimize import least_squares
import cv2

def rodrigues(r):
    return cv2.Rodrigues(r)[0]

def project_points(K, dist, R, t, points_3d):
    R_mat = rodrigues(R)
    t_vec = t.reshape(-1, 1)
    points_2d, _ = cv2.projectPoints(points_3d, R_mat, t_vec, K, dist)
    return points_2d.reshape(-1, 2)

def reprojection_error(params, n_cameras, n_points, camera_indices, point_indices, points_2d, K, dist):
    camera_params = params[:n_cameras * 6].reshape(n_cameras, 6)
    points_3d = params[n_cameras * 6:].reshape(n_points, 3)
    error = []
    for i in range(len(camera_indices)):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        R = camera_params[cam_idx, :3]
        t = camera_params[cam_idx, 3:6]
        projected = project_points(K[cam_idx], dist[cam_idx], R, t, points_3d[pt_idx:pt_idx+1])
        error.append(projected - points_2d[i])
    return np.concatenate(error).ravel()

def bundle_adjustment(K, dist, points_2d, camera_indices, point_indices, init_R, init_t, init_points_3d):
    n_cameras = len(K)
    n_points = len(np.unique(point_indices))
    
    camera_params = np.zeros((n_cameras, 6))
    for i in range(n_cameras):
        camera_params[i, :3] = cv2.Rodrigues(init_R[i])[0].ravel()
        camera_params[i, 3:6] = init_t[i].ravel()
    params = np.hstack((camera_params.ravel(), init_points_3d.ravel()))
    
    res = least_squares(
        reprojection_error,
        params,
        args=(n_cameras, n_points, camera_indices, point_indices, points_2d, K, dist),
        verbose=2
    )
    
    optimized_params = res.x
    optimized_camera_params = optimized_params[:n_cameras * 6].reshape(n_cameras, 6)
    optimized_points_3d = optimized_params[n_cameras * 6:].reshape(n_points, 3)
    
    optimized_R = [rodrigues(optimized_camera_params[i, :3]) for i in range(n_cameras)]
    optimized_t = optimized_camera_params[:, 3:6]
    
    return optimized_R, optimized_t, optimized_points_3d

if __name__ == "__main__":
    n_cameras = 3
    n_points = 10
    K = [np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]]) for _ in range(n_cameras)]
    dist = [np.zeros(5) for _ in range(n_cameras)] 
    points_2d = np.random.rand(30, 2) * 1280 
    camera_indices = np.repeat(np.arange(n_cameras), 10) 
    point_indices = np.tile(np.arange(n_points), n_cameras)
    init_R = [np.eye(3) for _ in range(n_cameras)] 
    init_t = [np.zeros(3) for _ in range(n_cameras)]
    init_points_3d = np.random.rand(n_points, 3) * 10
    
    print(len(points_2d))
    print(len(camera_indices))
    print(len(point_indices))
    print(len(init_points_3d))

    # optimized_R, optimized_t, optimized_points_3d = bundle_adjustment(
    #     K, dist, points_2d, camera_indices, point_indices, init_R, init_t, init_points_3d
    # )
    
    # for i in range(n_cameras):
    #     print(f"Camera {i+1} R:\n{optimized_R[i]}")
    #     print(f"Camera {i+1} t: {optimized_t[i]}")
    # print(f"Optimized 3D points:\n{optimized_points_3d}")