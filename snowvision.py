from scipy.spatial.transform import Rotation
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
import cv2
import numpy as np

class CameraGroup:
    class CamInfo:
        def __init__(self):
            self.K = np.zeros((3, 3))
            self.R = np.zeros((3, 3))
            self.t = np.zeros((3, 1))
            self.D = np.zeros((1, 5))

            self.original_K = np.zeros((3, 3))
            self.original_R = np.zeros((3, 3))
            self.original_t = np.zeros((3, 1))
            self.original_D = np.zeros((1, 5))

            self.Reprojection_Points = []
            self.Points = []
            self.Points_Ray = []

            self.HRNet_Reprojection_Points = []
            self.HRNet_Person_Points = []
            self.HRNet_Person_Ray = []
            self.HRNet_Person_Ray_Score = []

    def __init__(self, filepath, ax=None):
        self.Track_Points = []
        self.Load_Camera_Info(filepath, ax)

    def Load_Camera_Info(self, filepath, ax=None):
        self.Camera = []
        self.Camera_Count = 0
        with open(filepath) as f:
            lines = f.readlines()
            self.Camera_Count = int(lines[0].split('=')[1])
            print("Usable_Cam_Num：",self.Camera_Count)
            for i in range(self.Camera_Count):
                index = int(1 + i * 4)
                cam = self.CamInfo()
                K = Read_Array(lines[index + 1], (3, 3))
                Rt = Read_Array(lines[index + 2], (3, 4))
                D = Read_Array(lines[index + 3], (1, 5))
                cam.K = K
                cam.R = Rt[:,:3]
                cam.t = Rt[:,3].reshape(-1, 1)
                cam.D = D
                cam.original_K = cam.K
                cam.original_R = cam.R
                cam.original_t = cam.t
                cam.original_D = cam.D
                self.Camera.append(cam)
                print(lines[index])
                print("K：",K)
                print("Rt：", Rt)
                print("D：", D)      
                if ax != None:
                    Draw_Camera(cam.K, cam.R, cam.t, i, ax)

    def Camera_Translation(self, R_translate, t_translate, cam = -1):
        T_translate = R_and_t_to_T(R_translate, t_translate)
        if cam == -1:
            for c in range(self.Camera_Count):
                T = R_and_t_to_T(self.Camera[c].original_R, self.Camera[c].original_t)
                T_result = np.matmul(T_translate, T)
                R, t = T_to_R_and_t(T_result)
                self.Camera[c].R = R
                self.Camera[c].t = t
        else:
            T = R_and_t_to_T(self.Camera[cam].original_R, self.Camera[cam].original_t)
            T_result = np.matmul(T_translate, T)
            R, t = T_to_R_and_t(T_result)
            self.Camera[cam].R = R
            self.Camera[cam].t = t
        return T_translate
        
    def Draw_CameraGroup(self, ax, f=1):
        for i, cam in enumerate(self.Camera):
            Draw_Camera(cam.K, cam.R, cam.t, i, ax, f)

    def Add_Points_Ray(self, points, camera_index, ax = None):
        R = self.Camera[camera_index].R
        K = self.Camera[camera_index].K
        t = self.Camera[camera_index].t
        for p in points:
            w = np.array([p[0], p[1], 1])
            f = np.dot(np.linalg.inv(K), w)
            f = np.dot(R, f)
            if ax != None:
                a = f * 10
                ax.quiver(t[0], t[1], t[2], a[0], a[1], a[2])
            self.Camera[camera_index].Points_Ray.append(f.reshape((-1, 1)))
            self.Camera[camera_index].Points.append(p)

    def Triangulation(self, distance_tol, ax = None):
        self.Triangulate_Point = []
        for mc in range(self.Camera_Count - 1):
            tm = self.Camera[mc].t
            for sc in range(mc + 1, self.Camera_Count):
                ts = self.Camera[sc].t
                for hm in self.Camera[mc].Points_Ray:
                    for hs in self.Camera[sc].Points_Ray:
                        dist, W = Skew_Ray_Solver(hm, hs, tm, ts)
                        if dist < distance_tol:
                            self.Triangulate_Point.append(W)
        if ax != None:
            draw_TP = np.array(self.Triangulate_Point).T
            ax.scatter(draw_TP[0], draw_TP[1], draw_TP[2])

    def BA_Reprojecion_Error(self, params, W_list, w_list, c_list, p_list):
        param_len = 7
        r_index = 4
        cam_params = params[:self.Camera_Count * param_len].reshape((self.Camera_Count, param_len))
        cam_params = cam_params[1:]
        for i, param in enumerate(cam_params, start=1):
            q = param[:r_index]
            t = param[r_index:].reshape((-1, 1))
            self.Camera[i].R = Quaternion_to_Rotation_Matrix(q)
            self.Camera[i].t = t

        e = []
        for p, w, c in zip(p_list, w_list, c_list):
            Km = self.Camera[c].K
            Rm = self.Camera[c].R
            tm = self.Camera[c].t
            em = Reprojection_Error(W_list[p], Km, Rm, tm, w)
            e.append(em[0][0])
            e.append(em[1][0])         
        return e

    def BA_x0(self, W_list):
        x0 = np.array([])
        for i in range(self.Camera_Count):
            R = self.Camera[i].R
            q = Rotation_Matrix_to_Quaternion(R)
            t = self.Camera[i].t
            x0 = np.append(x0, q)
            x0 = np.append(x0, t)
        x0 = np.hstack((x0, np.array(W_list).ravel()))
        return x0

    def HRNet_Bundle_Adjustment(self, W_list, w_list, camera_indices, point_indices):
        J = bundle_adjustment_sparsity(self.Camera_Count, len(W_list), camera_indices, point_indices)
        x0 = self.BA_x0(W_list)
        e0 = self.BA_Reprojecion_Error(x0, W_list, w_list, camera_indices, point_indices)
        res = least_squares(self.BA_Reprojecion_Error, x0, jac_sparsity=J, verbose=2, ftol=1e-8, method='trf',
        args=(W_list, w_list, camera_indices, point_indices))
        e = self.BA_Reprojecion_Error(res['x'], W_list, w_list, camera_indices, point_indices)

        print("Before")
        print(np.sum(np.abs(e0)))
        print("After")
        print(np.sum(np.abs(e)))

    def Floor_Error(self, params, W_list):
        r_index = 4
        q = params[:r_index]
        t = params[r_index:].reshape((-1, 1))
        R = Quaternion_to_Rotation_Matrix(q)
        T = self.Camera_Translation(R, t)
        W_list_T = np.vstack((W_list.T, np.ones(len(W_list))))
        W_list_result = np.matmul(T, W_list_T)[2]

        return W_list_result

    def HRNet_Floor_Adjustment(self, W_list):
        x0 = np.array([1, 0, 0, 0, 0, 0, 0])
        e0 = self.Floor_Error(x0, W_list)
        res = least_squares(self.Floor_Error, x0, verbose=2, ftol=1e-8, method='trf', args=(W_list,))
        e = self.Floor_Error(res['x'], W_list)
        print("Before")
        print(np.sum(np.abs(e0)))
        print("After")
        print(np.sum(np.abs(e)))

    def Add_HRNet_Person_Ray(self, person, scores, camera_index, ax = None):
        R = self.Camera[camera_index].R
        K = self.Camera[camera_index].K
        t = self.Camera[camera_index].t
        person_ray_temp = []
        person_ray_score_temp = []
        
        for p, s in zip(person, scores):
            w = np.array([p[0], p[1], 1])
            f = np.dot(np.linalg.inv(K), w)
            f = np.dot(R, f)
            f = f.reshape((-1, 1))
            if ax != None:
                a = f * 10
                ax.quiver(t[0], t[1], t[2], a[0], a[1], a[2])
            person_ray_temp.append(f)
            person_ray_score_temp.append(s)
        self.Camera[camera_index].HRNet_Person_Points.append(person)
        self.Camera[camera_index].HRNet_Person_Ray.append(person_ray_temp)
        self.Camera[camera_index].HRNet_Person_Ray_Score.append(person_ray_score_temp)

    def HRNet_Triangulation(self, HRnet_score_weight = 1, tri_dist_weight = 1):
        self.HRNet_2DPoint = []
        self.HRNet_Camera_Index = []

        self.HRNet_Triangulate_Point = []
        self.HRNet_Triangulate_Point_Score = []
        self.HRNet_Triangulate_Point_Total_Score = []
        for mc in range(self.Camera_Count - 1):
            tm = self.Camera[mc].t
            for sc in range(mc + 1, self.Camera_Count):
                ts = self.Camera[sc].t
                for pm, pms, wm in zip(self.Camera[mc].HRNet_Person_Ray, self.Camera[mc].HRNet_Person_Ray_Score, self.Camera[mc].HRNet_Person_Points):
                    for ps, pss, ws in zip(self.Camera[sc].HRNet_Person_Ray, self.Camera[sc].HRNet_Person_Ray_Score, self.Camera[sc].HRNet_Person_Points):
                        p = []
                        pscore = []
                        for hm, hs, sm, ss in zip(pm, ps, pms, pss):
                            dist, W = Skew_Ray_Solver(hm, hs, tm, ts)
                            score = (1 - (sm + ss) / 2) * HRnet_score_weight + dist * tri_dist_weight
                            p.append(W)
                            
                            
                            pscore.append(score)
                        w0 = [wm, ws]
                        c = [mc, sc]

                        self.HRNet_Triangulate_Point.append(p)
                        self.HRNet_2DPoint.append(w0)
                        self.HRNet_Camera_Index.append(c)
                        self.HRNet_Triangulate_Point_Score.append(pscore) 
                        self.HRNet_Triangulate_Point_Total_Score.append(np.sum(pscore))

    def HRNet_Triangulation_Condense(self, distance_tol, reduce = True, condense_person_count_tol = 0, center_point_index = 19, keypoint_num = 20):
        tri_point_condense = []
        tri_point_condense_score = []
        tri_point_condense_total_score = []
        w0_condense = []
        cam_condense = []
        skip_list = []

        for center_point_main_index in range(len(self.HRNet_Triangulate_Point) - 1):
            if reduce:
                if (center_point_main_index in skip_list):
                    continue
            
            w0_condense_temp = []
            cam_condense_temp = []

            condense_person_count = 0
            center_point_main = self.HRNet_Triangulate_Point[center_point_main_index][center_point_index]
            tri_point_condense_temp = self.HRNet_Triangulate_Point[center_point_main_index]
            tri_point_condense_score_temp = self.HRNet_Triangulate_Point_Score[center_point_main_index]

            w0_condense_temp.append(self.HRNet_2DPoint[center_point_main_index])
            cam_condense_temp.append(self.HRNet_Camera_Index[center_point_main_index])

            for center_point_sub_index in range(center_point_main_index + 1, len(self.HRNet_Triangulate_Point)):
                center_point_sub = self.HRNet_Triangulate_Point[center_point_sub_index][center_point_index]

                dist = np.linalg.norm(center_point_main - center_point_sub)
                if dist < distance_tol:
                    skip_list.append(center_point_sub_index)
                    condense_person_count += 1
                    tri_point_condense_temp = np.append(tri_point_condense_temp, self.HRNet_Triangulate_Point[center_point_sub_index])
                    tri_point_condense_score_temp = np.append(tri_point_condense_score_temp, self.HRNet_Triangulate_Point_Score[center_point_sub_index])

                    w0_condense_temp.append(self.HRNet_2DPoint[center_point_sub_index])
                    cam_condense_temp.append(self.HRNet_Camera_Index[center_point_sub_index])

            w0_condense.append(w0_condense_temp)
            cam_condense.append(cam_condense_temp)

            if condense_person_count >= condense_person_count_tol:
                if len(tri_point_condense_temp) > (keypoint_num * 3):
                    tri_point_condense_temp = tri_point_condense_temp.reshape((len(tri_point_condense_temp) // (keypoint_num * 3), keypoint_num, 3))
                    tri_point_condense_score_temp = tri_point_condense_score_temp.reshape((len(tri_point_condense_score_temp) // keypoint_num, keypoint_num))
                    mean_points = np.mean(tri_point_condense_temp, axis=0)

                    p = []
                    ps = []
                    for i, s in enumerate(tri_point_condense_score_temp.T):
                        s_sum = np.sum(s)
                        s_rev_norm = np.array([])
                        non_zero_list = np.where(s != 0)[0]

                        if len(non_zero_list) == 0:
                            s_rev_norm = s

                        if len(non_zero_list) == 1:
                            s[non_zero_list] = 1
                            s_rev_norm = s

                        if len(non_zero_list) > 1:
                            s_rev = np.full((len(s)), s_sum) - s
                            remove_list = np.where(s_rev == s_sum)[0]
                            s_rev[remove_list] = 0
                            s_rev *= s_rev
                            s_rev_norm = s_rev / np.sum(s_rev)

                        condense_point = np.zeros(3)

                        for pn, sn in zip(tri_point_condense_temp[:, i], s_rev_norm):
                            condense_point += (pn - mean_points[i]) * sn
                        if np.sum(s_rev_norm) > 0:
                            condense_point += mean_points[i]

                        p.append(condense_point)
                        s_rev_norm[np.where(s_rev_norm == 0)[0]] = 10
                        ps.append(np.dot(s, s_rev_norm) / condense_person_count)

                    tri_point_condense.append(p)
                    tri_point_condense_score.append(ps)
                    tri_point_condense_total_score.append(np.sum(ps))

                else:
                    tri_point_condense.append(tri_point_condense_temp)
                    tri_point_condense_score.append(tri_point_condense_score_temp)
                    tri_point_condense_total_score.append(np.sum(tri_point_condense_score_temp))

        self.HRNet_Triangulate_Point = tri_point_condense
        self.HRNet_Triangulate_Point_Score = tri_point_condense_score
        self.HRNet_Triangulate_Point_Total_Score = tri_point_condense_total_score

        self.HRNet_2DPoint = w0_condense
        self.HRNet_Camera_Index = cam_condense

    def HRNet_Triangulation_Sort(self):
        w0 = []
        cam = []

        tri_point_sort = []
        tri_point_sort_score = []
        sort_list = sorted(self.HRNet_Triangulate_Point_Total_Score)

        for sl in sort_list:
            for p, s, ts, w, c in zip(self.HRNet_Triangulate_Point, 
                                      self.HRNet_Triangulate_Point_Score, 
                                      self.HRNet_Triangulate_Point_Total_Score, 
                                      self.HRNet_2DPoint,
                                      self.HRNet_Camera_Index):
                if sl == ts:
                    w0.append(w)
                    cam.append(c)
                    tri_point_sort.append(p)
                    tri_point_sort_score.append(s)
                    break

        self.HRNet_Triangulate_Point = tri_point_sort
        self.HRNet_Triangulate_Point_Score = tri_point_sort_score
        self.HRNet_Triangulate_Point_Total_Score = sort_list

        self.HRNet_2DPoint = w0
        self.HRNet_Camera_Index = cam

    def HRNet_Triangulation_ik(self):
        tri_point_ik = []

        for p in self.HRNet_Triangulate_Point:
            a = p[19] - p[18]
            b = p[12] - p[11]
            n_vec = np.cross(a, b)
            n_vec_nrom = n_vec / np.linalg.norm(n_vec)
            lower_spine_pole = p[18] + n_vec_nrom

            a = p[17] - p[19]
            b = p[6] - p[5]
            n_vec = np.cross(a, b)
            n_vec_nrom = n_vec / np.linalg.norm(n_vec)
            upper_spine_pole = p[17] + n_vec_nrom

            a = p[1] - p[0]
            b = p[2] - p[0]
            n_vec = np.cross(a, b)
            n_vec_nrom = n_vec / np.linalg.norm(n_vec)
            face_pole = p[0] + n_vec_nrom
            
            p.append(lower_spine_pole)
            p.append(upper_spine_pole)
            p.append(face_pole)
            tri_point_ik.append(p)

        self.HRNet_Triangulate_Point = tri_point_ik

    def HRNet_Triangulation_Tracking(self, distance_tol, max_tracking_num = 1, center_point_index = 19):
        if len(self.HRNet_Triangulate_Point) == 0:
            return        
        HRNet_triangulate_point_ref = np.array(self.HRNet_Triangulate_Point.copy())
        if self.Track_Points == []:#init trackpoint
            self.Track_Points = HRNet_triangulate_point_ref[:, center_point_index]
            if len(self.Track_Points) > max_tracking_num:#trim result to max_tracking_num
                self.Track_Points = self.Track_Points[:max_tracking_num]
                self.HRNet_Triangulate_Point = self.HRNet_Triangulate_Point[:max_tracking_num]
            return
        track_points_ref = HRNet_triangulate_point_ref[:, center_point_index]
        HRNet_triangulate_point_tracking = []
        track_points_temp = []
        for cp in self.Track_Points:
            ds = np.array([])
            for cpc in track_points_ref:
                ds = np.append(ds, np.linalg.norm(cpc - cp))
            if len(ds) == 0:
                break
            if np.min(ds) < distance_tol:
                ds_min_index = np.argmin(ds)
                HRNet_triangulate_point_tracking.append(HRNet_triangulate_point_ref[ds_min_index])
                track_points_temp.append(track_points_ref[ds_min_index])
                if len(track_points_ref) > 1:
                    track_points_ref = np.delete(track_points_ref, ds_min_index, 0)
                    HRNet_triangulate_point_ref = np.delete(HRNet_triangulate_point_ref, ds_min_index, 0)
        self.Track_Points = track_points_temp
        self.HRNet_Triangulate_Point = HRNet_triangulate_point_tracking
        #print("tracking", len(self.HRNet_Triangulate_Point))

    def Draw_Skeleton(self, ax, max_index = 0, isbone_label = False):
        bones = {
        'head_l':[0, 1], 
        'head_r':[0, 2], 
        'neck':[17, 0], 
        'shoulder_l':[17, 5], 
        'shoulder_r':[17, 6], 
        'arm_l':[5, 7], 
        'arm_r':[6, 8], 
        'elbow_l':[7, 9], 
        'elbow_r':[8, 10], 
        'chest':[19, 17], 
        'belly':[18, 19], 
        'hip_l':[18, 11], 
        'hip_r':[18, 12], 
        'thigh_l':[11, 13], 
        'thigh_r':[12, 14], 
        'calf_l':[13, 15], 
        'calf_r':[14, 16]}
        for i, person_s, in enumerate(zip(self.HRNet_Triangulate_Point, self.HRNet_Triangulate_Point_Score, self.HRNet_Triangulate_Point_Total_Score)):
            if i > max_index:
                break
            person = person_s[0]
            scores = person_s[1]
            total_scores = person_s[2]
            #ax.text(person[0][0], person[0][1], person[0][2], str(round(total_scores, 2)), size=10, zorder=1,  color='r')
            for b in bones:
                #print(scores[bones[b][0]], scores[bones[b][0]])
                if scores[bones[b][0]] == 0 or scores[bones[b][0]] == 0:
                    #print("skip")
                    continue
                bone_head = np.array([person[bones[b][0]][0], person[bones[b][0]][1], person[bones[b][0]][2]])
                bone_tail = np.array([person[bones[b][1]][0], person[bones[b][1]][1], person[bones[b][1]][2]])
                
                ax.plot3D(xs=(bone_head[0], bone_tail[0]),
                          ys=(bone_head[1], bone_tail[1]),
                          zs=(bone_head[2], bone_tail[2]))
                if isbone_label:
                    ax.text(bone_head[0], bone_head[1], bone_head[2], (b), size=5, zorder=1,  color='k')
            #for p, s in zip(person, scores):
            #    ax.text(p[0], p[1], p[2], str(round(s, 2)), size=10, zorder=1,  color='r')

    def Clear_Points_Ray(self):
        for camera_index in range(self.Camera_Count):
            self.Camera[camera_index].Reprojection_Points = []
            self.Camera[camera_index].Points = []
            self.Camera[camera_index].Points_Ray = []

            self.Camera[camera_index].HRNet_Reprojection_Points = []
            self.Camera[camera_index].HRNet_Person_Points = []
            self.Camera[camera_index].HRNet_Person_Ray = []
            self.Camera[camera_index].HRNet_Person_Ray_Score = []

    def OutputData(self, filename):
        with open(filename + '.txt', 'w') as f:
            f.write('Usable_Cam_Num=' + str(self.Camera_Count) + '\n')
            for num in range(self.Camera_Count):
                f.write('Cam_' + str(num) + ':\n')
                K_temp = self.Camera[num].K.reshape((9))
                f.write('K=')
                for k in K_temp:
                    f.write(str(float(k)) + ',')
                f.write('\n')
                Rt = np.hstack((self.Camera[num].R, self.Camera[num].t))
                Rt_temp = Rt.reshape((12))
                f.write('Rt=')
                for Rt in Rt_temp:
                    f.write(str(float(Rt)) + ',')
                f.write('\n')
                D_temp = self.Camera[num].D[0]
                f.write('D=')
                for D in D_temp:
                    f.write(str(float(D)) + ',')
                print(K_temp)
                print(Rt_temp)
                print(D_temp)
                f.write('\n')

def Read_Array(line, shape):
        A = np.array([])
        K_str = line.split('=')[1].split(',')
        for i in range(int(shape[0] * shape[1])):
            k_temp = float(K_str[i])
            A = np.append(A, k_temp)
        A = A.reshape(shape)
        return A

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

def Load_Video(video_name, usable_cam_num):
    cap_array = np.array([])
    img_size_array = np.array([])
    length_array = np.array([], dtype=int)
    for num in range(usable_cam_num):
        cap_temp = cv2.VideoCapture(video_name + str(num) + '.avi')
        length_temp = int(cv2.VideoCapture.get(cap_temp, cv2.CAP_PROP_FRAME_COUNT))
        v_w = int(cv2.VideoCapture.get(cap_temp, cv2.CAP_PROP_FRAME_WIDTH))
        v_h = int(cv2.VideoCapture.get(cap_temp, cv2.CAP_PROP_FRAME_HEIGHT))
        img_size_temp = np.array([v_w, v_h])
        cap_array = np.append(cap_array, cap_temp)
        length_array = np.append(length_array, length_temp)
        img_size_array = np.append(img_size_array, img_size_temp)
        print('Get video : ' + video_name + str(num) + '.avi')
    print('Video resolution : ' + str(img_size_array))
    print('Video total frame number : ' + str(length_array))
    img_size_array = img_size_array.reshape((usable_cam_num, 2))
    return cap_array, length_array, img_size_array

def Load_Image(image_name, usable_cam_num):
    img_array = np.array([])
    img_size_array = np.array([])
    for num in range(usable_cam_num):
        img_temp = cv2.imread(image_name + str(num) + '.png')
        v_w = img_temp.shape[1]
        v_h = img_temp.shape[0]
        img_size_temp = np.array([v_w, v_h])
        img_array = np.append(img_array, img_temp)
        img_size_array = np.append(img_size_array, img_size_temp)
        print('Get image : ' + image_name + str(num) + '.png')
        print('image resolution : ' + str(img_size_array))
    img_size_array = img_size_array.reshape((usable_cam_num, 2))
    return img_array, img_size_array

def Find_Usable_Cam(cam_search_limit):
    usable_cam = np.array([])
    for num in range(cam_search_limit):
        video_temp = cv2.VideoCapture(num)
        ret, _ = video_temp.read()
        if ret:
            usable_cam = np.append(usable_cam, num)
            globals()['cap' + str(num)] = cv2.VideoCapture(num)
        else:
            break
    for num in range(len(usable_cam)):
        video_temp = globals()['cap' + str(num)]
        ret, frame_temp = video_temp.read()
        if np.any(frame_temp) == None:
            usable_cam = np.delete(usable_cam, num)
    return usable_cam

def K_From_F(K1,K2,R,T):
    A = K1 * R.T * T
    C = np.array([0,-A[2],A[1]],[A[2],0,-A[0]],[-A[1],A[0],0])
    F = np.linalg.inv(K2).T * R * K1 * C
    return F

def F_From_P(P1,P2):
    X = np.zeros((3,2,4))
    X[0] = np.array([P1[1],P1[2]])
    X[1] = np.array([P1[2],P1[0]])
    X[2] = np.array([P1[0],P1[1]])

    Y = np.zeros((3, 2, 4))
    Y[0] = np.array([P2[1],P2[2]])
    Y[1] = np.array([P2[2],P2[0]])
    Y[2] = np.array([P2[0],P2[1]])

    F = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            XY = np.array([X[j][0],X[j][1],Y[i][0],Y[i][1]])
            F[i][j] = np.linalg.det(XY)
    return F

def R_and_t_to_T(R, t):
    T = np.hstack((R, t))
    T = np.vstack((T, [0, 0, 0, 1]))
    return T

def T_to_R_and_t(T):
    Rt = T[:3]
    R = Rt[:, :3]
    t = Rt[:, 3].reshape((-1, 1))
    return R, t
    
def Rotation_Matrix_to_Quaternion(R):
    r = Rotation.from_matrix(R)
    return r.as_quat()

def Quaternion_to_Rotation_Matrix(q):
    r = Rotation.from_quat(q)
    return r.as_matrix()

def Rotation_Matrix_to_Rotation_Vector(R):
    r = Rotation.from_matrix(R)
    return r.as_rotvec()

def Rotation_Vector_to_Rotation_Matrix(v):
    r = Rotation.from_rotvec(v)
    return r.as_matrix()

def Binary_2DPoints(processed_image, min_area = 0, max_area = 70):
    points = np.array([])
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for num, cnt in enumerate(contours):
        M = cv2.moments(cnt)
        a = cv2.contourArea(cnt)
        if a > min_area and a < max_area:
            xdot = M["m10"] / M["m00"]
            ydot = M["m01"] / M["m00"]
            points_temp = np.array([xdot, ydot])
            points = np.append(points, points_temp)
    points = np.reshape(points,(len(points)//2,2))
    return points

def Point_Group_Label(image, points, islocation_text = True, offset = 10, color = (255, 0, 0)):
    for i, p in enumerate(points):
        x = int(p[0])
        y = int(p[1])
        cv2.circle(image, (x, y), 5, color, -1)
        if islocation_text:
            cv2.putText(image, str(i) + str((x, y)), (x, y - offset), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 0), 2)

def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 7 + n_points * 3
    A = lil_matrix((m, n), dtype=int)
    i = np.arange(camera_indices.size)
    for s in range(7):
        A[2 * i, camera_indices * 7 + s] = 1
        A[2 * i + 1, camera_indices * 7 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 7 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 7 + point_indices * 3 + s] = 1
    return A

def Reprojection(W, K, R, t):
    W = W.reshape((-1, 1))
    R_inv = np.linalg.inv(R)
    W_T = np.matmul(R_inv, W - t)
    sw = np.matmul(K, W_T)
    s = sw[2]
    w = sw / s
    w = w.reshape(3)
    return w[:2], sw

def Reprojection_Error(W, K, R, t, w):
    W = W.reshape((-1, 1))
    R_inv = np.linalg.inv(R)
    W_T = np.matmul(R_inv, W - t)
    sw = np.matmul(K, W_T)
    s = sw[2]
    w_re = sw / s
    w_re = w_re.reshape(3)[:2]
    e = w - w_re
    return e.reshape((-1, 1))

def Skew_Ray_Solver(hm, hs, tm, ts):
    H = np.hstack((hm, hs))
    ne1 = np.linalg.inv(np.dot(H.T, H))
    ne2 = np.dot(H.T, (ts - tm))
    S = np.dot(ne1, ne2)
    Wm = hm * S[0] + tm
    Ws = -hs * S[1] + ts
    return np.linalg.norm(Wm - Ws), ((Wm + Ws) / 2).reshape(3)