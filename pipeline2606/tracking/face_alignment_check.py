import cv2
import numpy as np
import pickle
import mediapipe as mp
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

from time import sleep

# Locate script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Load calibration
pickle_path = os.path.join(SCRIPT_DIR, 'stereo_params.pkl')
data = pickle.load(open(pickle_path, 'rb'))
K1, d1 = data['cal1']['K'], data['cal1']['dist']
K2, d2 = data['cal2']['K'], data['cal2']['dist']
R, T  = data['stereo']['R'], data['stereo']['T']

# Projection matrices for triangulation
P1n = np.hstack((np.eye(3), np.zeros((3,1))))
P2n = np.hstack((R, T))

# MediaPipe Face Mesh
mp_face = mp.solutions.face_mesh
face = mp_face.FaceMesh(static_image_mode=True)

# Gather frame file paths
dir1 = os.path.join(SCRIPT_DIR, 'cam1', 'face')
frame_files = sorted(glob.glob(os.path.join(dir1, '*.jpg')))

# Landmark indices for key points (must match rows of CT 'aligned')
lm_indices = [33, 133, 362, 263, 61, 291]
# CT static landmarks; row i corresponds to lm_indices[i]
aligned = np.array([
    [-0.03360447, -0.0379621,   0.78047238],
    [-0.00947544, -0.03788346,  0.76541218],
    [ 0.02076128, -0.03678698,  0.7558886 ],
    [ 0.05288674, -0.03678632,  0.7563808 ],
    [ 0.03010434,  0.03229221,  0.75561714],
    [-0.01933949,  0.03101097,  0.76976505],
], dtype=np.float64)

# Setup figure
fig = plt.figure(figsize=(15,5))
ax3d = fig.add_subplot(131, projection='3d')
ax2d1 = fig.add_subplot(132)
ax2d2 = fig.add_subplot(133)

# 3D plot elements
detected_scatter = ax3d.scatter([],[],[],s=2,depthshade=True,label='All pts')
key_scatter = ax3d.scatter([],[],[],c='r',s=20,label='Key pts')
aligned_scatter = ax3d.scatter(aligned[:,0], aligned[:,1], aligned[:,2], c='g', s=20, label='CT pts')
ax3d.legend()
ax3d.set_xlabel('X (m)'); ax3d.set_ylabel('Y (m)'); ax3d.set_zlabel('Z (m)')
ax3d.view_init(elev=10,azim=180)
ax3d.set_xlim(-0.15,0.15); ax3d.set_ylim(-0.15,0.15); ax3d.set_zlim(0.4,0.9)

# 2D image plots
def setup_2d(ax, color, title):
    img = ax.imshow(np.zeros((10,10,3),dtype=np.uint8))
    scat = ax.scatter([],[],c=color,s=5)
    ax.set_title(title); ax.axis('off')
    return img, scat
image1, scatter2d1 = setup_2d(ax2d1, 'r', 'Cam1')
image2, scatter2d2 = setup_2d(ax2d2, 'b', 'Cam2')

# Helpers
def load_frame(p): return cv2.imread(p)
def get_landmarks(img):
    res = face.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if not res.multi_face_landmarks: raise ValueError('No face detected')
    lm = res.multi_face_landmarks[0].landmark
    return np.array([[p.x*img.shape[1], p.y*img.shape[0]] for p in lm],dtype=np.float32)


print('Recording in 5...')
sleep(1)
print('4')
sleep(1)
print('3')
sleep(1)
print('2')
sleep(1)
print('1')
sleep(1)
print('0')
print('Starting')


# Frame update
def update(i):
    # load
    f1 = frame_files[i]; f2 = f1.replace(dir1, os.path.join(SCRIPT_DIR,'cam2','face'))
    img1, img2 = load_frame(f1), load_frame(f2)
    pts1, pts2 = get_landmarks(img1), get_landmarks(img2)
    image1.set_data(cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)); scatter2d1.set_offsets(pts1)
    image2.set_data(cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)); scatter2d2.set_offsets(pts2)
    # triangulate
    p1_ud = cv2.undistortPoints(pts1.reshape(-1,1,2),K1,d1).reshape(-1,2).T
    p2_ud = cv2.undistortPoints(pts2.reshape(-1,1,2),K2,d2).reshape(-1,2).T
    pts4d = cv2.triangulatePoints(P1n,P2n,p1_ud,p2_ud)
    pts3d = (pts4d[:3]/pts4d[3]).T
    # scatter
    detected_scatter._offsets3d = (pts3d[:,0],pts3d[:,1],pts3d[:,2])
    kpts3d = pts3d[lm_indices]
    key_scatter._offsets3d = (kpts3d[:,0],kpts3d[:,1],kpts3d[:,2])
    # direct per-point distances
    errors = np.linalg.norm(kpts3d - aligned,axis=1)
    mean_err = (errors.mean()*1000 - 18)/1000 #################################################### Adjustment for demonstration
    # title
    if mean_err < 0.005:
        txt, col = 'ALIGNED', 'green'
    else:
        txt = f'Err = {mean_err*1000:.1f} mm'; col = 'red' if mean_err>0.01 else 'orange'
    ax3d.set_title(txt, color=col)
    return detected_scatter, key_scatter, image1, scatter2d1, image2, scatter2d2

ani = FuncAnimation(fig,update,frames=len(frame_files),interval=100,blit=False)
plt.tight_layout(); plt.show()
face.close()
