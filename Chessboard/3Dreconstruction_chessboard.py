import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



video=False
image=True
#Calibration parameters
square_size = 0.025  #Square size in [m]
chessboard_dims = (8, 6)  #Chessboard size (internal corners)

def extract_calibration_information(camera_path):
    config_file = camera_path + '/config.xml'
    fs = cv2.FileStorage(config_file, cv2.FILE_STORAGE_READ)
    camera_matrix = fs.getNode("CameraMatrix").mat()
    distortion_coeffs = fs.getNode("DistortionCoeffs").mat()
    rotation_vector = fs.getNode("Rotation").mat()
    translation_vector = fs.getNode("Translation").mat()
    fs.release()
    return camera_matrix, distortion_coeffs, rotation_vector, translation_vector

def compute_reprojection_error(points_3d, corners, camera_matrix, dist_coeffs, rvec, tvec):
    total_error = 0
    projected_points, _ = cv2.projectPoints(
        np.array([points_3d[i] for i in sorted(points_3d.keys())]),
        rvec, tvec, camera_matrix, dist_coeffs
    )

    projected_points = projected_points.reshape(-1, 2)
    actual_points = corners.reshape(-1, 2)

    errors = np.linalg.norm(actual_points - projected_points, axis=1)
    mean_error = np.mean(errors)

    print(f"Reprojection error: {mean_error:.4f} pixels")
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

def triangulate_all_points(corners1, corners2,camera_matrix_cam1, distortion_coeffs_cam1,camera_matrix_cam2, distortion_coeffs_cam2, P1, P2):
    points_3d_result = {}
    for i in range(len(corners1)):
        p1 = np.array([[corners1[i][0][0]], [corners1[i][0][1]]], dtype=float)
        p2 = np.array([[corners2[i][0][0]], [corners2[i][0][1]]], dtype=float)

        point_3d_h = cv2.triangulatePoints(P1, P2, p1, p2)
        point_3d = point_3d_h[:3] / point_3d_h[3]
        points_3d_result[i] = point_3d.flatten()
    return points_3d_result

def calculate_square_distances(points_3d):
    points_array = np.array([points_3d[i] for i in sorted(points_3d.keys())])
    rows = 6
    cols = 8
    grid = points_array.reshape((rows, cols, 3))  # 3 for x, y, z
    
    horizontal_distances = np.linalg.norm(grid[:, 1:, :] - grid[:, :-1, :], axis=2)
    vertical_distances = np.linalg.norm(grid[1:, :, :] - grid[:-1, :, :], axis=2)
    
    mean_horizontal = horizontal_distances.mean()
    mean_vertical = vertical_distances.mean()

    #print(mean_horizontal)
    #print(mean_vertical)
    
    error= abs(mean_horizontal - square_size*1000) / abs(square_size*1000)
    error= abs(mean_vertical - square_size*1000) / abs(square_size*1000)
    total_error= abs(mean_horizontal + mean_vertical) / abs(2)
    total_relative_error = total_error / abs(square_size*1000)

    #print(f"Absolut mean error: {error:.4f} mm")
    #print(f'Relative mean error: {total_relative_error: .4f} %')
    return total_error, total_relative_error 

def plot_3d_landmarks_variant(points_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #Lists to store the values of the 3D points
    x_vals, y_vals, z_vals, labels = [], [], [], []

    #Add 3D points to the lists
    for idx, coords in points_3d.items():
        x_vals.append(coords[0])
        y_vals.append(coords[1])
        z_vals.append(coords[2])
        labels.append(f'Punto {idx}')

    # ------------------ VISUALISATION OF THE 3D POINTS ------------------------------

    ax.scatter(x_vals, y_vals, z_vals, color='r', s=30)
    #To label points
    for i, label in enumerate(labels):
        ax.text(x_vals[i], y_vals[i], z_vals[i], label, color='blue', fontsize=10, verticalalignment='bottom')

    # Show origin (0,0,0) in the graph 
    ax.scatter(0, 0, 0, color='g', s=100, label="Origin (0, 0, 0)", marker='x')
    ax.text(0, 0, 0, "Origin (0, 0, 0)", color='green', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    #Adjust 3D view
    ax.view_init(elev=20, azim=-60)
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([-200, 200])
            
    plt.title('Puntos 3D del Chessboard')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)


def plot_3d_landmarks_variant2(points_3d):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Lists to store the values of the 3D points
    x_vals, y_vals, z_vals, labels = [], [], [], []

    # Add 3D points to the lists
    for idx, coords in points_3d.items():
        x_vals.append(coords[0])
        y_vals.append(coords[1])
        z_vals.append(coords[2])
        #labels.append(f'Punto {idx}')

    # Plot the 3D points
    ax.scatter(x_vals, y_vals, z_vals, color='r', s=30)

    # Label each point
    for i, label in enumerate(labels):
        ax.text(x_vals[i], y_vals[i], z_vals[i], label, color='blue', fontsize=10, verticalalignment='bottom')

    # Show origin
    ax.scatter(0, 0, 0, color='g', s=100, label="Origen (0, 0, 0)", marker='x')
    ax.text(0, 0, 0, "Origen (0, 0, 0)", color='green', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

    # Draw axis arrows
    axis_len = 100  # Length of each axis line
    ax.quiver(0, 0, 0, axis_len, 0, 0, color='r', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color='g', arrow_length_ratio=0.1)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color='b', arrow_length_ratio=0.1)

    # Label axes
    ax.text(axis_len, 0, 0, 'X', color='red', fontsize=12)
    ax.text(0, axis_len, 0, 'Y', color='green', fontsize=12)
    ax.text(0, 0, axis_len, 'Z', color='blue', fontsize=12)

    # Adjust 3D view
    ax.view_init(elev=20, azim=-60)
    ax.set_xlim([-200, 200])
    ax.set_ylim([-200, 200])
    ax.set_zlim([-200, 200])

    ax.set_xlabel("Eje X")
    ax.set_ylabel("Eje Y")
    ax.set_zlabel("Eje Z")

    plt.title('Puntos 3D del Chessboard')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
    cv2.waitKey(0)

def plot_3d_landmarks_video(points_3d):
    
    ax.clear()
    ax.set_title("3D Chessboard Points (Real-Time)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    xs = [pt[0] for pt in points_3d.values()] #he cambiado la z por la x
    ys = [pt[1] for pt in points_3d.values()]
    zs = [pt[2] for pt in points_3d.values()]
    
    ax.scatter(xs, ys, zs, c='b', marker='o')
    ax.set_xlim([-300, 1000])
    ax.set_ylim([-400, 1000])
    ax.set_zlim([-700, 700])
    
    plt.draw()
    plt.pause(0.001)  # Permite la actualización sin bloqueo

def swap_yz(points_3d):
    swapped_points = {}
    for idx, (x, y, z) in points_3d.items():
        swapped_points[idx] = (x, z, y)  # Intercambia Y y Z
    return swapped_points

def translate_points(points_3d, reference_point):
    """
    Translates a dictionary of 3D points so that the reference_point becomes the origin.
    
    Args:
        points_3d (dict): Dictionary with keys as point IDs and values as (x, y, z) coordinates.
        reference_point (tuple): The reference point (x, y, z) to translate all points around.

    Returns:
        dict: A new dictionary with translated 3D points.
    """
    translated_points = {}

    for idx, (x, y, z) in points_3d.items():
        translated_points[idx] = (
            x - reference_point[0],
            y - reference_point[1],
            z - reference_point[2]
        )

    return translated_points

#Camara configuration
camera_path_cam1 = './data_trial/cam1'
camera_path_cam2 = './data_trial/cam2'

camera_matrix_cam1, distortion_coeffs_cam1, rot_vec1, trans_vec1 = extract_calibration_information(camera_path_cam1)
camera_matrix_cam2, distortion_coeffs_cam2, rot_vec2, trans_vec2 = extract_calibration_information(camera_path_cam2)

P1 = get_projection_matrix(camera_matrix_cam1, rot_vec1, trans_vec1)
P2 = get_projection_matrix(camera_matrix_cam2, rot_vec2, trans_vec2)

if image:
    #Read images
    chessboard1 = cv2.imread(r".\data_trial\cam1\chess.jpg")
    chessboard2 = cv2.imread(r".\data_trial\cam2\chess.jpg")



    #Detection of chessboard corners for both cameras
    corners1 = find_chessboard_corners(chessboard1, chessboard_dims)
    corners2 = find_chessboard_corners(chessboard2, chessboard_dims)

    if corners1 is not None and corners2 is not None:
        #Draw selected corners in the image
        cv2.drawChessboardCorners(chessboard1, chessboard_dims, corners1, True)
        cv2.drawChessboardCorners(chessboard2, chessboard_dims, corners2, True)

        #TRIANGULATION STEP
        points_3d = triangulate_all_points(corners1, corners2,camera_matrix_cam1, distortion_coeffs_cam1, camera_matrix_cam2, distortion_coeffs_cam2, P1, P2 )

        #Show images with the detected points
        cv2.imshow("Chessboard 1", chessboard1)
        cv2.imshow("Chessboard 2", chessboard2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        print("\n--- 3D Points ---")
        for idx, coords in points_3d.items():
            print(f"Point {idx}: {coords}")


        print("\n--- Reprojection Errors ---")
        compute_reprojection_error(points_3d, corners1, camera_matrix_cam1, distortion_coeffs_cam1, rot_vec1, trans_vec1)
        compute_reprojection_error(points_3d, corners2, camera_matrix_cam2, distortion_coeffs_cam2, rot_vec2, trans_vec2)

        points_3d = swap_yz(points_3d)

        total_error, total_relative_error =calculate_square_distances(points_3d)
        print('total error: ', total_error,'  total_relative_error: ', total_relative_error)
        #plot_3d_landmarks_variant(points_3d)
        plot_3d_landmarks_variant2(points_3d)
        cv2.waitKey(0)

       
        reference = (50, 100, 150)

        translated = translate_points(points_3d, reference)
        plot_3d_landmarks_variant2(translated)
        cv2.waitKey(0)

    else:
        print("The corners of the board were not found in the images")

if video:
    #READ VIDEOS
    plt.ion()  # Modo interactivo activado
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    best_error = float('inf')
    worst_error = float('-inf')
    best_frame_points = None
    worst_frame_points = None
    error_log = []

    #chessboard1_video = cv2.VideoCapture(r".\data_trial\cam1\intrinsics.mp4")
    #chessboard2_video = cv2.VideoCapture(r".\data_trial\cam2\intrinsics.mp4")

    chessboard1_video = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    chessboard2_video = cv2.VideoCapture(2, cv2.CAP_DSHOW)

    while True:
        # Leer un frame de cada cámara
        ret1, frame1 = chessboard1_video.read()
        ret2, frame2 = chessboard2_video.read()

        if not ret1 or not ret2:
            print("No se pudieron leer más frames.")
            break

        # Buscar esquinas en ambos frames
        corners1 = find_chessboard_corners(frame1, chessboard_dims)
        corners2 = find_chessboard_corners(frame2, chessboard_dims)

        if corners1 is not None and corners2 is not None:
            # Dibujar esquinas
            cv2.drawChessboardCorners(frame1, chessboard_dims, corners1, True)
            cv2.drawChessboardCorners(frame2, chessboard_dims, corners2, True)

            # Triangulación de puntos 3D
            points_3d = triangulate_all_points(corners1, corners2,
                                            camera_matrix_cam1, distortion_coeffs_cam1,
                                            camera_matrix_cam2, distortion_coeffs_cam2,
                                            P1, P2)

            error_mm, relative_error = calculate_square_distances(points_3d)
            error_log.append(error_mm)

            if error_mm < best_error:
                best_error = error_mm
                best_frame_points = points_3d

            if error_mm > worst_error:
                worst_error = error_mm
                worst_frame_points = points_3d

            points_3d = swap_yz(points_3d)

            plot_3d_landmarks_video(points_3d)
       

        #plot_3d_landmarks_video(points_3d)
    # Liberar recursos
    chessboard1_video.release()
    chessboard2_video.release()
    cv2.destroyAllWindows()

        # Mostrar resumen
    print("\n--- RESULTADOS DEL VIDEO ---")
    print(f"Mejor error de distancia (mínimo): {best_error:.2f} mm")
    print(f"Peor error de distancia (máximo): {worst_error:.2f} mm")