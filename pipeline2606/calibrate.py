'''
stereo_calibration.py

Performs intrinsic calibration for two cameras, stereo calibration, and saves parameters.

Usage: python stereo_calibration.py
'''

import cv2
import numpy as np
import glob
import pickle
import os

# Locate script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Chessboard parameters
tile_size = 0.033625  # meters (3.3625 cm)
pattern_size = (7, 4)  # internal corners (width, height)

# Prepare object points: (0,0,0), (1,0,0), ... scaled by tile_size
def create_object_points(pattern_size, tile_size):
    objp = np.zeros((pattern_size[1]*pattern_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
    objp *= tile_size
    return objp

objp = create_object_points(pattern_size, tile_size)

# Calibrate one camera: collect intrinsics images
def calibrate_camera(image_folder):
    objpoints, imgpoints = [], []
    images = glob.glob(f"{image_folder}/*.jpg")
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                 cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                        (cv2.TermCriteria_EPS + cv2.TermCriteria_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
    # calibrate
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # compute reprojection error
    tot_error = 0
    for i in range(len(objpoints)):
        imgpts2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        error = cv2.norm(imgpoints[i], imgpts2, cv2.NORM_L2)/len(imgpts2)
        tot_error += error
    mean_error = tot_error/len(objpoints)
    return {'ret': ret, 'K': K, 'dist': dist, 'rvecs': rvecs, 'tvecs': tvecs, 'reproj_error': mean_error}

# Calibrate cam1 and cam2 intrinsics
cam1_intrinsics_dir = os.path.join(SCRIPT_DIR, 'cam1/intrinsics')
cam2_intrinsics_dir = os.path.join(SCRIPT_DIR, 'cam2/intrinsics')
cal1 = calibrate_camera(cam1_intrinsics_dir)
cal2 = calibrate_camera(cam2_intrinsics_dir)
print(f"Cam1 reproj error: {cal1['reproj_error']:.4f} m")
print(f"Cam2 reproj error: {cal2['reproj_error']:.4f} m")

# Stereo calibration: use simultaneous extrinsics images
def stereo_calibrate(folder1, folder2, cal1, cal2):
    objpoints, imgpts1, imgpts2 = [], [], []
    imgs1 = sorted(glob.glob(f"{folder1}/*.jpg"))
    imgs2 = sorted(glob.glob(f"{folder2}/*.jpg"))
    for f1, f2 in zip(imgs1, imgs2):
        i1 = cv2.imread(f1); g1 = cv2.cvtColor(i1, cv2.COLOR_BGR2GRAY)
        i2 = cv2.imread(f2); g2 = cv2.cvtColor(i2, cv2.COLOR_BGR2GRAY)
        r1, c1 = cv2.findChessboardCorners(g1, pattern_size, None)
        r2, c2 = cv2.findChessboardCorners(g2, pattern_size, None)
        if r1 and r2:
            objpoints.append(objp)
            c1s = cv2.cornerSubPix(g1, c1, (11,11), (-1,-1), (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER,30,0.001))
            c2s = cv2.cornerSubPix(g2, c2, (11,11), (-1,-1), (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER,30,0.001))
            imgpts1.append(c1s); imgpts2.append(c2s)
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpts1, imgpts2,
        cal1['K'], cal1['dist'], cal2['K'], cal2['dist'], g1.shape[::-1],
        criteria=(cv2.TermCriteria_MAX_ITER+cv2.TermCriteria_EPS, 100, 1e-5), flags=flags)
    print(f"Stereo calibration RMS error: {ret:.4f}")
    return {'R': R, 'T': T, 'E': E, 'F': F, 'rms': ret}

# Calibrate cam1 and cam2 extrinsics
cam1_extrinsics_dir = os.path.join(SCRIPT_DIR, 'cam1/extrinsics')
cam2_extrinsics_dir = os.path.join(SCRIPT_DIR, 'cam2/extrinsics')

stereo = stereo_calibrate(cam1_extrinsics_dir, cam2_extrinsics_dir, cal1, cal2)

# Save all calibration data
pickle_path = os.path.join(SCRIPT_DIR, 'stereo_params.pkl')
with open(pickle_path, 'wb') as f:
    pickle.dump({'cal1': cal1, 'cal2': cal2, 'stereo': stereo}, f)

print('Calibration parameters saved to stereo_params.pkl')
