import cv2
import numpy as np
import matplotlib.pyplot as plt

# 讀取左右圖像
img_left = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('2.png', cv2.IMREAD_GRAYSCALE)

# 選擇特徵檢測器：ORB 或 SIFT
use_orb = True  # 設為 False 使用 SIFT

if use_orb:
    # 初始化 ORB 檢測器
    detector = cv2.ORB_create(nfeatures=1000)
else:
    # 初始化 SIFT 檢測器
    detector = cv2.SIFT_create()

# 檢測特徵點和計算描述子
keypoints_left, descriptors_left = detector.detectAndCompute(img_left, None)
keypoints_right, descriptors_right = detector.detectAndCompute(img_right, None)

# 創建匹配器
if use_orb:
    # ORB 使用 Hamming 距離
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
else:
    # SIFT 使用 L2 範數
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

# 進行特徵點匹配
matches = matcher.match(descriptors_left, descriptors_right)

# 按距離排序匹配結果（距離越小越好）
matches = sorted(matches, key=lambda x: x.distance)

# 選取前 N 個最佳匹配
num_matches = 200
matches = matches[:num_matches]

# 提取匹配點的坐標
points_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
points_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 可視化匹配結果
img_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, matches, None, 
                              flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 顯示匹配結果
plt.figure(figsize=(15, 5))
plt.imshow(img_matches, cmap='gray')
plt.title('Feature Matches')
plt.axis('off')
plt.show()

# 如果需要計算基礎矩陣（Fundamental Matrix）來過濾錯誤匹配
F, mask = cv2.findFundamentalMat(points_left, points_right, cv2.FM_RANSAC)

# 篩選內點（inliers）
inlier_matches = [matches[i] for i in range(len(matches)) if mask[i]]

# 可視化篩選後的匹配
img_inlier_matches = cv2.drawMatches(img_left, keypoints_left, img_right, keypoints_right, inlier_matches, None,
                                     flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# 顯示篩選後的匹配結果
plt.figure(figsize=(15, 5))
plt.imshow(img_inlier_matches, cmap='gray')
plt.title('Inlier Matches after RANSAC')
plt.axis('off')
plt.show()