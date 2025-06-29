# -----
# face3d_projection.py
# Reads stereo_params.pkl, detects face landmarks with MediaPipe, and triangulates points.


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
R, T = data['stereo']['R'], data['stereo']['T']

# Projection matrices
P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
P2 = K2 @ np.hstack((R, T))

# MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=True)

# Read images
img1_dir = os.path.join(SCRIPT_DIR, 'cam1/face/frame_00030.jpg')
img2_dir = os.path.join(SCRIPT_DIR, 'cam2/face/frame_00030.jpg')
img1 = cv2.imread(img1_dir)
img2 = cv2.imread(img2_dir)
h1, w1 = img1.shape[:2]; h2, w2 = img2.shape[:2]

# Detect landmarks (use first detection)
def get_landmarks(img):
    res = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks:
        raise ValueError('No face detected')
    lm = res.multi_face_landmarks[0].landmark
    pts = np.array([[p.x*img.shape[1], p.y*img.shape[0]] for p in lm], dtype=np.float32)
    return pts

pts1 = get_landmarks(img1)
pts2 = get_landmarks(img2)

# Undistort points
pts1_ud = cv2.undistortPoints(pts1.reshape(-1,1,2), K1, d1)
pts2_ud = cv2.undistortPoints(pts2.reshape(-1,1,2), K2, d2)

# Identity
print('Moving onto identity')
P1n = np.hstack((np.eye(3), np.zeros((3,1))))   # 3×4
P2n = np.hstack((   R,            T   ))       # 3×4

# flatten to shape (2,N)
print('Flattening')
x1 = pts1_ud.reshape(-1,2).T
x2 = pts2_ud.reshape(-1,2).T

# Triangulate
print('Triangulating')
pts4d = cv2.triangulatePoints(P1n, P2n, x1, x2)
pts3d = (pts4d[:3] / pts4d[3]).T  # now truly in **meters**, since T is meters



# 3D plot:
print('3D plotting')
fig = plt.figure(figsize=(8, 6))
ax  = fig.add_subplot(111, projection='3d')

# scatter all points
ax.scatter(
    pts3d[:, 0],    # X coordinates
    pts3d[:, 1],    # Y coordinates
    pts3d[:, 2],    # Z coordinates
    s=2,            # marker size
    depthshade=True
)

# Label a few key landmarks in a different color
key_idxs = [1, 33, 61, 199]
nose_idx = 4     # MediaPipe’s primary nose‐tip landmark :contentReference[oaicite:0]{index=0}
chin_idx = 152   # Landmark at bottom of chin :contentReference[oaicite:1]{index=1}
key_idxs = [nose_idx, chin_idx]
ax.scatter(
    pts3d[key_idxs, 0],
    pts3d[key_idxs, 1],
    pts3d[key_idxs, 2],
    c='r',
    s=20,
    label='nose tip, eye, etc.'
)

# Extract the 3D coordinates
nose_pt = pts3d[nose_idx]
chin_pt = pts3d[chin_idx]

# Euclidean distance (in meters)
dist = np.linalg.norm(nose_pt - chin_pt)
print(f"Distance nose→chin: {dist:.3f} m")

# Axis labels
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D Face Landmarks from Stereo Triangulation')
ax.legend(loc='best')

# Optional: equal aspect ratio
max_range = (pts3d.max(axis=0) - pts3d.min(axis=0)).max() / 2.0
mid_x = (pts3d[:,0].max() + pts3d[:,0].min()) * 0.5
mid_y = (pts3d[:,1].max() + pts3d[:,1].min()) * 0.5
mid_z = (pts3d[:,2].max() + pts3d[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

plt.show()

# Optionally visualize
# for p in pts3d:
#     cv2.circle(img1, (int(p[0]), int(p[1])), 2, (0,255,0), -1)

# Release MediaPipe
face.close()
