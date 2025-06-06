import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import mediapipe as mp
matplotlib.use('Qt5Agg') 
import pandas as pd

landmark_names = {
        33: "Left exocanthion",
        133: "Left endocanthion",
        6: "Nasion",
        362: "Right endocanthion",
        263: "Right exocanthion",
        4: "Pronasale",
        98: "Left alar crest",
        327: "Right alar crest",
        2: "Subnasale",
        61: "Left cheilion",
        291: "Right cheilion",
        0: "Labiale superius (outer)",
        17: "Labiale inferius (outer)",
        199: "Pogonion" 
    }


def extract_calibration_information(camera_path):
    config_file = camera_path + '/config_100frames.xml'
    fs = cv2.FileStorage(config_file, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("CameraMatrix").mat()
    distortion_coeffs = fs.getNode("DistortionCoeffs").mat()
    rotation_vector = fs.getNode("Rotation").mat()
    translation_vector = fs.getNode("Translation").mat()
    fs.release()
    return camera_matrix, distortion_coeffs, rotation_vector, translation_vector

def compute_reprojection_error(points_3d, corners, camera_matrix, dist_coeffs, rvec, tvec):
    projected_points, _ = cv2.projectPoints(
        np.array([points_3d[i] for i in sorted(points_3d.keys())]),
        rvec, tvec, camera_matrix, dist_coeffs
    )

    projected_points = projected_points.reshape(-1, 2)
    actual_points = corners.reshape(-1, 2)

    errors = np.linalg.norm(actual_points - projected_points, axis=1)
    mean_error = np.mean(errors)

    return mean_error

def get_projection_matrix(camera_matrix, rotation_vector, translation_vector):
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    Rt = np.hstack((rotation_matrix, translation_vector))
    return camera_matrix @ Rt

def find_chessboard_corners(image, chessboard_dims):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, chessboard_dims, None)
    if ret:
        corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        return corners
    return None

def triangulate_all_points(corners1, corners2, P1, P2):
    if len(corners1) != len(corners2):
        print(f"Not same points detected! points1 = {len(corners1)}, points2 = {len(corners2)}")

    else:
        points_3d_result = {}
        for i in range(len(corners1)):
            p1 = np.array([[corners1[i][0][0]], [corners1[i][0][1]]], dtype=float)
            p2 = np.array([[corners2[i][0][0]], [corners2[i][0][1]]], dtype=float)

            point_3d_h = cv2.triangulatePoints(P1, P2, p1, p2)
            point_3d = point_3d_h[:3] / point_3d_h[3]
            points_3d_result[i] = point_3d.flatten()
        return points_3d_result

def calculate_square_distances(points_3d, square_size):
    points_array = np.array([points_3d[i] for i in sorted(points_3d.keys())])
    rows = 6
    cols = 8
    grid = points_array.reshape((rows, cols, 3)) 
    
    horizontal_distances = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2)
    vertical_distances = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2)
    
    mean_horizontal = horizontal_distances.mean()
    mean_vertical = vertical_distances.mean()

    total_distance= abs(mean_horizontal + mean_vertical) / abs(2)
    #print(total_distance)
    total_error= abs(abs(square_size*1000)- total_distance)
    total_relative_error = total_error / abs(square_size*1000)

    return total_distance, total_relative_error 

def plot_3d_landmarks_variant(*pointset):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for i, (points, color_chess) in enumerate(pointset):
    # Lists to store the values of the 3D points
        x_vals, y_vals, z_vals, labels = [], [], [], []
        
        if points is not None:
        # Add 3D points to the lists
            for idx, coords in points.items():
                x_vals.append(coords[0])
                y_vals.append(coords[1])
                z_vals.append(coords[2])
          
        # Plot the 3D points
        ax.scatter(x_vals, y_vals, z_vals, color = color_chess, s=10)

        # Label each point
        for i, label in enumerate(labels):
            ax.text(x_vals[i], y_vals[i], z_vals[i], label, color='blue', fontsize=10, verticalalignment='bottom')

    # Show origin
    ax.scatter(0, 0, 0, color='g', s=100, label="(0, 0, 0)", marker='x')

    # Draw axis arrows
    axis_len = 100  # Length of each axis line
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', arrow_length_ratio=0.1)

    # Label axes
    ax.text(axis_len, 0, 0, 'X', color='red', fontsize=12)
    ax.text(0, axis_len, 0, 'Y', color='green', fontsize=12)
    ax.text(0, 0, axis_len, 'Z', color='blue', fontsize=12)

    # # Adjust 3D view
    ax.view_init(elev=-77, azim=-90)  
    # ax.set_xlim([0, 900])
    # ax.set_ylim([0, 900])
    # ax.set_zlim([-200, 200])

    plt.title(f'3D Points')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)


def mean_3d_distance(dict1, dict2):
    assert dict1.keys() == dict2.keys(), "Both dictionaries must have the same keys"
    
    distances = []
    for key in dict1:
        point1 = dict1[key]
        point2 = dict2[key]
        dist = np.linalg.norm(point1 - point2)
        distances.append(dist)
    
    mean_distance = np.mean(distances)
    return mean_distance
    
def find_facial_landmarks(image):
    
    points_list = []
    #Facial landmarks
    result = face_mesh.process(image)

    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            for idx, lm in enumerate(facial_landmarks.landmark):
                if idx in landmark_names:
                    h, w, _ = image.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    points_list.append([[x, y]]) 
        
    points_array = np.array(points_list, dtype=np.float32)  
    return points_array

if __name__ == "__main__":

    video=False
    real_time=False
    image=True

    chessboard =False
    face = True

    #Face mesh
    mp_face_mesh= mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    #Calibration parameters
    square_size = 0.023  #Square size in [m]
    chessboard_dims = (8, 6)  #Chessboard size (internal corners)

    distances = [2, 4, 6, 8]

    for distance in distances:

        print(f"-----------DISTANCES   {distance}-------------")
        #Camara configuration
        camera_path_cam1 = f".\\New experiment\\{distance}\\cam1"
        camera_path_cam2 =  f".\\New experiment\\{distance}\\cam2"

        camera_matrix_cam1, distortion_coeffs_cam1, rot_vec1, trans_vec1 = extract_calibration_information(camera_path_cam1)
        camera_matrix_cam2, distortion_coeffs_cam2, rot_vec2, trans_vec2 = extract_calibration_information(camera_path_cam2)

        P1 = get_projection_matrix(camera_matrix_cam1, rot_vec1, trans_vec1)
        P2 = get_projection_matrix(camera_matrix_cam2, rot_vec2, trans_vec2)

        if image:

            chessboard_image_names = ["extrinsics.jpg", "1.jpg", "2.jpg", "3.jpg"]
            face_image_names = ["5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg"] 

            corners_cam1 = []
            corners_cam2 = []
            face_landmarks_cam1 = []
            face_landmarks_cam2 = []

            triangulated_chessboard_points = []
            triangulated_face_points = []


            for name in chessboard_image_names:
                img1 = cv2.imread(f"{camera_path_cam1}\\{name}")
                img2 = cv2.imread(f"{camera_path_cam2}\\{name}")

                c1 = find_chessboard_corners(img1, chessboard_dims)
                c2 = find_chessboard_corners(img2, chessboard_dims)

                corners_cam1.append(c1)
                corners_cam2.append(c2)

                pts3d = triangulate_all_points(c1, c2, P1, P2)
                triangulated_chessboard_points.append(pts3d)
            
            for name in face_image_names:
                img1 = cv2.imread(f"{camera_path_cam1}\\{name}")
                img2 = cv2.imread(f"{camera_path_cam2}\\{name}")

                l1 = find_facial_landmarks(img1)
                l2 = find_facial_landmarks(img2)

                face_landmarks_cam1.append(l1)
                face_landmarks_cam2.append(l2)

                pts3d = triangulate_all_points(l1, l2, P1, P2)
                triangulated_face_points.append(pts3d)

            reprojection_error_cam1 = []
            reprojection_error_cam2 = []

            for pts_3d, corners1, corners2 in zip(triangulated_chessboard_points, corners_cam1, corners_cam2):
                reprojection_error_cam1.append(
                    compute_reprojection_error(pts_3d, corners1, camera_matrix_cam1, distortion_coeffs_cam1, rot_vec1, trans_vec1)
                )
                reprojection_error_cam2.append(
                    compute_reprojection_error(pts_3d, corners2, camera_matrix_cam2, distortion_coeffs_cam2, rot_vec2, trans_vec2)
                )

            mean_reprojection_error_cam1 = np.mean(reprojection_error_cam1)
            mean_reprojection_error_cam2 = np.mean(reprojection_error_cam2)

            print(f"Mean reprojection error cam 1: {mean_reprojection_error_cam1:.4f} mm, from this individual reprojection errors: {reprojection_error_cam1}")
            print(f"Mean reprojection error cam 2: {mean_reprojection_error_cam2:.4f} mm, from this individual reprojection errors: {reprojection_error_cam2}")

           # print(len(triangulated_chessboard_points))

            if chessboard:      
                real_distances = [310, 540, 500] 
                if len(triangulated_chessboard_points) >= 2:
                    for i in range(len(triangulated_chessboard_points) - 1):
                        points_a = triangulated_chessboard_points[i]
                        points_b = triangulated_chessboard_points[i + 1]

                        # Calcular distancia media 3D
                        mean_dist = mean_3d_distance(points_a, points_b)
                        print(f"[{i+1}→{i+2}] Mean 3D distance: {mean_dist:.2f} mm")

                        # Si tenemos una distancia real para comparar, calcular errores
                        if i < len(real_distances):
                            real_dist = real_distances[i]
                            error_abs = abs(mean_dist - real_dist)
                            error_rel = error_abs / real_dist * 100
                            print(f"[{i+1}→{i+2}] Absolute error: {error_abs:.2f} mm, Relative error: {error_rel:.2f}%")
                else:
                    print("Not enough  point sets to compute distances.")

                    for i, points in enumerate(triangulated_chessboard_points[0:]):
                        total_distance, total_relative_error = calculate_square_distances(points, square_size)
                        print(f"Image {i+1} - Mean chessboard distance: {total_distance:.2f} mm, Relative error: {total_relative_error:.2%}")

            if face:    
                if len(triangulated_face_points) >= 2:
                    for i in range(len(triangulated_face_points) - 1):
                        
                        points_a = triangulated_face_points[i]
                        points_b = triangulated_face_points[i + 1]
                        if points_a is not None and points_b is not None:
                            if len(points_a) == len(points_b):
                                #Calcular distancia media 3D
                                mean_dist = mean_3d_distance(points_a, points_b)
                                print(f"[{i+1}→{i+2}] Mean 3D distance: {mean_dist:.2f} mm")

                else:
                    print("Not enough point sets to compute distances.")

            landmark_sets = []

            #To show the chessboard points
            if chessboard:
                chessboard_colors = ['red', 'blue', 'green', 'yellow']
                for pts, col in zip(triangulated_chessboard_points, chessboard_colors):
                    landmark_sets.append((pts, col))

            #To show the face points
            if face: 
                face_colors = ['red', 'blue', 'green', 'magenta']
                for i, pts in enumerate(triangulated_face_points):
                    color = face_colors[i % len(face_colors)]
                    landmark_sets.append((pts, color))

            #Show 3D points
            plot_3d_landmarks_variant(*landmark_sets)
            cv2.waitKey(0)
