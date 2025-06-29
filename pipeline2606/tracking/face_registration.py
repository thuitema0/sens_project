import cv2
import numpy as np
import pickle
import mediapipe as mp
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers the 3D projection

# Locate script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load calibration
pickle_path = os.path.join(SCRIPT_DIR, 'stereo_params.pkl')
data = pickle.load(open(pickle_path, 'rb'))
K1, d1 = data['cal1']['K'], data['cal1']['dist']
K2, d2 = data['cal2']['K'], data['cal2']['dist']
R, T  = data['stereo']['R'], data['stereo']['T']

# Projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K2 @ np.hstack((R, T))

# MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face   = mp_face.FaceMesh(static_image_mode=True)

# Read images
def load_frame(cam, frame='frame_00540.jpg'):
    return cv2.imread(os.path.join(SCRIPT_DIR, cam, 'face', frame))
img1, img2 = load_frame('cam1'), load_frame('cam2')

# Detect landmarks
def get_landmarks(img):
    res = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError('No face detected')
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x*img.shape[1], p.y*img.shape[0]] for p in lm], dtype=np.float32)

pts1, pts2 = get_landmarks(img1), get_landmarks(img2)

# Undistort points
pts1_ud = cv2.undistortPoints(pts1.reshape(-1,1,2), K1, d1).reshape(-1,2).T
pts2_ud = cv2.undistortPoints(pts2.reshape(-1,1,2), K2, d2).reshape(-1,2).T

# Identity (for triangulation)
P1n = np.hstack((np.eye(3), np.zeros((3,1))))
P2n = np.hstack((R, T))

# Triangulate
pts4d = cv2.triangulatePoints(P1n, P2n, pts1_ud, pts2_ud)
pts3d = (pts4d[:3] / pts4d[3]).T  # X,Y,Z in meters

# ---- RIGID (or similarity) REGISTRATION ----
# Define corresponding landmark indices in pts3d:
# left eye outer, left eye inner, right eye inner, right eye outer, left mouth corner, right mouth corner
lm_indices = [33, 133, 362, 263, 61, 291]
fixed_pts = pts3d[lm_indices]  # shape (6,3)

# CT scan (moving) landmarks (in same order)
'''
moving_pts = np.array([
    [-41.45, -297.66, 123.86],  # left eye outer
    [-14.12, -305.54, 123.81],  # left eye inner
    [ 17.58, -306.39, 123.07],  # right eye inner
    [ 48.34, -297.14, 123.81],  # right eye outer
    [ 27.49, -301.40,  20.25],  # right mouth corner
    [-23.95, -301.36,  20.90],  # left mouth corner
    #[ 27.49, -301.40,  54.25],  # right mouth corner
    #[-23.95, -301.36,  54.90],  # left mouth corner
], dtype=np.float64)
'''

# CT scan (moving) landmarks (in same order)
moving_pts = np.array([
    [-41.45, -297.66, 123.86],  # left eye outer
    [-14.12, -305.54, 123.81],  # left eye inner
    [ 17.58, -306.39, 123.07],  # right eye inner
    [ 48.34, -297.14, 123.81],  # right eye outer
    [ 27.49, -301.40,  54.25],  # right mouth corner
    [-23.95, -301.36,  54.90],  # left mouth corner
], dtype=np.float64)

moving_pts = moving_pts/1000

# Compute similarity transform (Umeyama)
def compute_similarity(src, dst, with_scaling=True):
    # src, dst: (N,3)
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src
    dst_c = dst - mu_dst
    # covariance
    H = src_c.T @ dst_c / src.shape[0]
    U, S, Vt = np.linalg.svd(H)
    R_ = Vt.T @ U.T
    # ensure right-handed
    if np.linalg.det(R_) < 0:
        Vt[-1,:] *= -1
        R_ = Vt.T @ U.T
    if with_scaling:
        var_src = (src_c**2).sum() / src.shape[0]
        scale = (S.sum() / var_src)
    else:
        scale = 1.0
    t_ = mu_dst - scale * R_ @ mu_src
    return scale, R_, t_

# Toggle scaling True/False
use_scaling = False
s, R_reg, t_reg = compute_similarity(moving_pts, fixed_pts, with_scaling=use_scaling)
# Transform CT points
aligned = (s * (R_reg @ moving_pts.T)).T + t_reg
print(aligned)

# 3D plot
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection='3d')
# original cloud
ax.scatter(pts3d[:,0], pts3d[:,1], pts3d[:,2], s=2, depthshade=True, label='Reconstructed')
# aligned CT
ax.scatter(aligned[:,0], aligned[:,1], aligned[:,2], c='r', s=20, label='Aligned CT')
ax.legend()
# set labels/aspect
ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
max_range = np.ptp(pts3d, axis=0).max()/2
mid = pts3d.mean(axis=0)
ax.set_xlim(mid[0]-max_range, mid[0]+max_range)
ax.set_ylim(mid[1]-max_range, mid[1]+max_range)
ax.set_zlim(mid[2]-max_range, mid[2]+max_range)
plt.show()

# Release MediaPipe
face.close()
