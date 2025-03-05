import numpy as np
import math

class SecondOrderDynamic:
    def __init__(self, f, z, r, x0):
        pi = math.pi
        self.k1 = z / (pi * f)
        self.k2 = 1 / ((2 * pi * f) * (2 * pi * f))
        self.k3 = r * z / (2 * pi * f)

        self.xp = x0
        self.y = x0
        self.yd = 0

    def update(self, T, x, xd = None):
        if(xd == None):
            xd = (x - self.xp) / T
            self.xp = x

        self.y = self.y + T * self.yd
        self.yd = self.yd + T * (x + self.k3 * xd - self.y - self.k1 * self.yd) / self.k2
        return self.y

def Skew_Ray_Solver(hm, hs, tm, ts):
    H = np.hstack((hm, hs))
    ne1 = np.linalg.inv(np.dot(H.T, H))
    ne2 = np.dot(H.T, (ts - tm))
    S = np.dot(ne1, ne2)
    Wm = hm * S[0] + tm
    Ws = -hs * S[1] + ts
    return np.linalg.norm(Wm - Ws), ((Wm + Ws) / 2).reshape(3)

def Triangulation(camera_group, distance_tol, ax = None):
    Triangulate_Point = []
    for mc in range(camera_group.camera_num - 1):
        tm = camera_group.cameras[mc].t
        for sc in range(mc + 1, camera_group.camera_num):
            ts = camera_group.cameras[sc].t
            for hm in camera_group.cameras[mc].Points_Ray:
                for hs in camera_group.cameras[sc].Points_Ray:
                    dist, W = Skew_Ray_Solver(hm, hs, tm, ts)
                    if dist < distance_tol:
                        Triangulate_Point.append(W)
    if ax != None:
        draw_TP = np.array(Triangulate_Point).T
        ax.scatter(draw_TP[0], draw_TP[1], draw_TP[2])
    
    return Triangulate_Point

def Human_Triangulation(camera_group, keypoint_score_threshold=0.5, average_score_threshold=0.0, distance_threshold=0.05):
    #hrnet_2d_points = []
    #hrnet_camera_indice = []
    hrnet_triangulate_points = []
    hrnet_triangulate_keypoint_scores = []
    hrnet_triangulate_person_scores = []
    for mc in range(camera_group.camera_num - 1):
        tm = camera_group.cameras[mc].t
        for sc in range(mc + 1, camera_group.camera_num):
            ts = camera_group.cameras[sc].t
            for pm, pms, pwm in zip(camera_group.cameras[mc].hrnet_point_rays, 
                                   camera_group.cameras[mc].hrnet_point_score, 
                                   camera_group.cameras[mc].hrnet_points):
                for ps, pss, pws in zip(camera_group.cameras[sc].hrnet_point_rays, 
                                       camera_group.cameras[sc].hrnet_point_score, 
                                       camera_group.cameras[sc].hrnet_points):
                    p_3d_points = []
                    p_2d_points = []
                    p_camera_indice = []
                    p_score = []
                    for hm, hs, sm, ss, wm, ws in zip(pm, ps, pms, pss, pwm, pws):
                        dist, W = Skew_Ray_Solver(hm, hs, tm, ts)
                        score = ((sm + ss) / 2) / (dist * 1000)
                        if sm < keypoint_score_threshold or ss < keypoint_score_threshold or dist > distance_threshold:
                            score = 0
                        p_3d_points.append(W)
                        p_2d_points.append([wm, ws])
                        p_camera_indice.append([mc, sc])
                        p_score.append(score)
                    p_score_avg_score = np.mean(p_score)
                    if p_score_avg_score < average_score_threshold:
                        continue

                    hrnet_triangulate_points.append(np.array(p_3d_points))
                    #hrnet_2d_points.append(np.array(p_2d_points))
                    #hrnet_camera_indice.append(np.array(p_camera_indice))
                    hrnet_triangulate_keypoint_scores.append(np.array(p_score))
                    hrnet_triangulate_person_scores.append(p_score_avg_score) 

    result = {'hrnet_triangulate_points' : hrnet_triangulate_points,
              'hrnet_triangulate_keypoint_scores' : hrnet_triangulate_keypoint_scores,
              'hrnet_triangulate_person_scores' : hrnet_triangulate_person_scores}

    return result

def Human_Triangulation_Condense(result, 
                                 condense_distance_tol = 0.1, 
                                 condense_person_num_tol = 0, 
                                 condense_score_tol = 0.0, 
                                 center_point_index = 18, 
                                 keypoint_num = 30):
    person_num = len(result['hrnet_triangulate_points'])

    condensed_person_list = []
    condensed_person_keypoint_scores_list = []
    condensed_person_scores_list = []
    condensed_person_index_list = []
    for mc in range(person_num - 1):
        if mc in condensed_person_index_list:
            continue
        
        main_person = result['hrnet_triangulate_points'][mc]
        main_center_point = main_person[center_point_index]
        main_person_keypoint_scores = result['hrnet_triangulate_keypoint_scores'][mc]
        condense_person_list = [main_person]
        condense_person_scores_list = [main_person_keypoint_scores]
        for sc in range(mc + 1, person_num):
            if sc in condensed_person_index_list:
                continue

            sub_person = result['hrnet_triangulate_points'][sc]
            sub_center_point = sub_person[center_point_index]
            sub_person_keypoint_scores = result['hrnet_triangulate_keypoint_scores'][sc]

            dist = np.linalg.norm(main_center_point - sub_center_point)
            if dist > condense_distance_tol:
                continue
            
            condensed_person_index_list.append(sc)
            condense_person_list.append(sub_person)
            condense_person_scores_list.append(sub_person_keypoint_scores)
        
        condense_person_num = len(condense_person_list)
        if condense_person_num < condense_person_num_tol:
            continue
        
        condense_person = np.zeros((keypoint_num, 3))
        condense_person_keypoint_scores = np.zeros(keypoint_num)
        for body_part in range(keypoint_num):
            condense_body_part_scores = np.array([condense_person_scores_list[idx][body_part] 
                                                  for idx in range(condense_person_num)])
            sum_score = np.sum(condense_body_part_scores)
            if sum_score == 0.0:
                continue
            condense_body_part_scores = condense_body_part_scores / sum_score
            condense_body_part_list = np.array([condense_person_list[idx][body_part] * condense_body_part_scores[idx]
                                           for idx in range(condense_person_num)])
            condense_person[body_part] = np.sum(condense_body_part_list, axis=0)
            condense_person_keypoint_scores[body_part] = sum_score / condense_person_num

        avg_score = np.mean(condense_person_keypoint_scores)
        if avg_score < condense_score_tol:
            continue

        condensed_person_list.append(condense_person)
        condensed_person_keypoint_scores_list.append(condense_person_keypoint_scores)
        condensed_person_scores_list.append(avg_score)

    result = {'hrnet_triangulate_points' : condensed_person_list,
              'hrnet_triangulate_keypoint_scores' : condensed_person_keypoint_scores_list,
              'hrnet_triangulate_person_scores' : condensed_person_scores_list}
    
    return result
    
def Human_Triangulation_Smooth(result, previous_result = None, f = 2, z = 0.75, r = 0, delta_time = 1 / 30):
    damped_skeleton_list = []
    second_order_dynamic_list = []
    
    if isinstance(previous_result, dict):
        for person, sod in zip(result['hrnet_triangulate_points'], previous_result['second_order_dynamics']):
            damped_skeleton = [pp[1].update(delta_time, pp[0]) for pp in zip(person, sod)]
            damped_skeleton_list.append(damped_skeleton)
        
        result = {'hrnet_triangulate_points' : damped_skeleton_list,
                  'hrnet_triangulate_keypoint_scores' : result['hrnet_triangulate_keypoint_scores'],
                  'hrnet_triangulate_person_scores' : result['hrnet_triangulate_person_scores'],
                  'second_order_dynamics' : previous_result['second_order_dynamics']}
    else:
        for person in result['hrnet_triangulate_points']:
            second_order_dynamic_list.append([SecondOrderDynamic(f, z, r, p) for p in person])

        result = {'hrnet_triangulate_points' : result['hrnet_triangulate_points'],
                  'hrnet_triangulate_keypoint_scores' : result['hrnet_triangulate_keypoint_scores'],
                  'hrnet_triangulate_person_scores' : result['hrnet_triangulate_person_scores'],
                  'second_order_dynamics' : second_order_dynamic_list}

    return result


