import numpy as np
import cv2 as cv
import os


calibrate_intrinsics = False # extrinsics calibration included
calibrate_extrinsics_only = False  # intrinsics calibration not included
show_axes = False
show_chessboard = False
analysis_frames=True

#read checkerboard config variables
checkerboard_handle = cv.FileStorage('./data_trial/checkerboard.xml', cv.FILE_STORAGE_READ)
board_width = int(checkerboard_handle.getNode('CheckerBoardWidth').real())
board_height = int(checkerboard_handle.getNode('CheckerBoardHeight').real())
tile_size = checkerboard_handle.getNode('CheckerBoardSquareSize').real()
checkerboard_handle.release()

# manual annotation of chessboard corners in case it is not automatically detected
def manual_annotation(window_name, image, board_size):
    #initialize 2d corner structure
    corners = []
    
    #set mouse callback
    cv.setMouseCallback(window_name, click, (corners, window_name, image))
    
    #corners should start left top, and run clockwise
    while len(corners) < 4:
        cv.waitKey(50)

    #transform chessboard area to rectangle
    (width, height) = board_size
    transform_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    perspective_matrix = cv.getPerspectiveTransform(np.array(corners, dtype=np.float32), transform_points)
    inverse_perspective_matrix = np.linalg.pinv(perspective_matrix)

    #interpolate chessboard corners 
    grid_coordinates = np.zeros((width * height, 2), np.float32)
    grid_coordinates[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    return cv.perspectiveTransform(np.array([[grid_point] for grid_point in grid_coordinates], np.float32), inverse_perspective_matrix)


#mouse click event handler
def click(event_name, x, y, flags, event_data):
    (corners, window_name, image) = event_data
    if event_name == cv.EVENT_LBUTTONDOWN:
        corners.append([x, y])

        # draw circle
        cv.circle(image, (x, y), 3, (255, 255, 0), -1)
        cv.imshow(window_name, image)

#intrinsic camera calibration from a video with chessboard
def calibrate_intrinsics_from_video(file_name,frame_step):

    file_handle = cv.VideoCapture(file_name)
    if not file_handle.isOpened():
        print(f"The video file could not be opened:: {file_name}")
        return None, None 

    num_frames = int(file_handle.get(cv.CAP_PROP_FRAME_COUNT))
    #print(f"Total number of frames: {num_frames}")

    #initialize image list
    image_list = []

    #loop over all frames with step size of (number to decide: before was 50)
    for i_image in range(0, num_frames, frame_step):
        ret, frame = file_handle.read()
        if ret:
            image_list.append(frame)
            
    # 3d world coordinates of chessboard cornders
    chessboard_3d_coordinates = np.zeros((board_width * board_height, 3), np.float32)
    chessboard_3d_coordinates[:, :2] = tile_size * np.mgrid[0:board_width, 0:board_height].T.reshape(-1,2)

    # initialize arrays for 2d and 3d chessboard corner coordinates
    corner_2d_points = []
    corner_3d_points = []

    # loop over chessboard images
    for image in image_list:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #print(gray_image.shape[::-1])

        if gray_image is None:
            print("Error: Could not convert image to grayscale. ")
            continue

        # attempt to find the chessboard corners
        ret, found_corners = cv.findChessboardCorners(gray_image, (board_width, board_height), None)
        #print(ret)

        if ret:
            # corners found, refine
            refined_corners = cv.cornerSubPix(gray_image, found_corners, (7, 7), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        else:
            continue #Now, if the corners are not found automatically, we don't use the manual annotation, we skip the frame (more accurate results)
            # corners not found, proceed to manual annotation
            cv.imshow('manual_calibration', image)
            annotated_corners = manual_annotation('manual_calibration', image, (board_width, board_height))
            refined_corners = cv.cornerSubPix(gray_image, annotated_corners, (1, 1), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            cv.destroyAllWindows()

        # add 2d and 3d chessboard corner coordinates
        corner_2d_points.append(refined_corners)
        corner_3d_points.append(chessboard_3d_coordinates)

        # display image with chessboard corners
        if show_chessboard:
            cv.drawChessboardCorners(image, (board_width, board_height), refined_corners, ret)
            cv.imshow('Chessboard', image)
            cv.waitKey(500)

    #CALIBRATE CAMERA
    print(f"Number of 2D points detected: {len(corner_2d_points)}")
    print(f"Number of 3D points detected: {len(corner_3d_points)}")
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(corner_3d_points, corner_2d_points, gray_image.shape[::-1], None, None)
    
    #REPROJECTION ERROR
    mean_error = 0
    for i in range(len(corner_3d_points)):
        imgpoints2, _ = cv.projectPoints(corner_3d_points[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(corner_2d_points[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        mean_error += error

    print( "total error: {}".format(mean_error/len(corner_3d_points)) )

    return mtx, dist


def calibrate_intrinsics_from_folder(folder_path, board_width, board_height, tile_size, frame_step):
    # Get and sort image file names in the folder
    image_files = sorted([
        os.path.join(folder_path, f) for f in os.listdir(folder_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
    ])

    if not image_files:
        print(f"No images found in folder: {folder_path}")
        return None, None

    image_list = []

    # Select images with the specified step
    for i in range(0, len(image_files), frame_step):
        image = cv.imread(image_files[i])
        if image is not None:
            image_list.append(image)

    print(f"Selected {len(image_list)} images (with step {frame_step})")

    # Prepare 3D chessboard corner coordinates
    chessboard_3d_coordinates = np.zeros((board_width * board_height, 3), np.float32)
    chessboard_3d_coordinates[:, :2] = tile_size * np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

    corner_2d_points = []
    corner_3d_points = []

    # Detect chessboard corners in each selected image
    for image in image_list:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(gray, (board_width, board_height))

        if found:
            corner_2d_points.append(corners)
            corner_3d_points.append(chessboard_3d_coordinates)

    print(f"Detected corners {len(corner_2d_points)}")

    # If enough corners were found, perform calibration
    if len(corner_2d_points) > 0:
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
            corner_3d_points, corner_2d_points, gray.shape[::-1], None, None)
        print(f"Mean reprojection error: {ret:.4f} pixels")
        return camera_matrix, dist_coeffs, len(image_list), ret
    else:
        print("Not enough corner detections for calibration.")
        return None, None


#extrinsic camera calibration from the selected image with chessboard
def calibrate_extrinsics_from_image(image_path, mtx, dist):
    # Cargar la imagen
    frame = cv.imread(image_path)
    gray_image = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    if frame is None:
        raise FileNotFoundError(f"The image could not be loaded: {image_path}")

    #3D world coordinates (corners of the checkerboard)
    chessboard_3d_coordinates = np.zeros((board_width * board_height, 3), np.float32)
    chessboard_3d_coordinates[:, :2] = tile_size * np.mgrid[0:board_width, 0:board_height].T.reshape(-1, 2)

     # attempt to find the chessboard corners
    ret, found_corners = cv.findChessboardCorners(gray_image, (board_width, board_height), None)
        #print(ret)
    if ret:
        # corners found, refine
        corners = cv.cornerSubPix(gray_image, found_corners, (7, 7), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
    else:
        #Show the image for manual annotation
        cv.imshow("Imagen de calibración", frame)
        corners = manual_annotation("Imagen de calibración", frame, (board_width, board_height))

    #Solve extrinsics parameters
    ret, rvec, tvec = cv.solvePnP(chessboard_3d_coordinates, corners, mtx, dist)

    cv.destroyAllWindows()
    return rvec, tvec

#main loop of the program.
def main():
    reprojection_results = {f"cam{i+1}": {} for i in range(2)}  # Define before loop if using in both cases
    frame_step_list = [2, 5, 10, 20, 50, 100]
    #frame_step_list = [50,100 ]

    # Configure all cameras
    for i in range(2):
        camera_name = f"cam{i+1}"
        camera_path = r'.\data_trial\cam' + str(i+1)
        print(camera_path)
        folder_path=os.path.join(camera_path, 'intrinsics_frames')
        print(folder_path)
        # ================================================================
        # calibrate intrinsics & extrinsics, or read calibration from file
        # ================================================================

        if calibrate_intrinsics:
            # calibrate intrinsics
            (mtx, dist) = calibrate_intrinsics_from_video(camera_path + '\\intrinsics.mp4',10)
 
            # calibrate extrinsics
            (rvec, tvec) = calibrate_extrinsics_from_image(camera_path + '\\chess.jpg', mtx, dist)

            # save calibration
            file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_WRITE)
            file_handle.write('CameraMatrix', mtx)
            file_handle.write('DistortionCoeffs', dist)
            file_handle.write('Rotation', rvec)
            file_handle.write('Translation', tvec)
            file_handle.release()

        elif analysis_frames:

            for frame_step in frame_step_list:
                mtx, dist, num_frames, reproj_error = calibrate_intrinsics_from_folder(
                    folder_path, board_width, board_height, tile_size, frame_step)

                if reproj_error is not None:
                    reprojection_results[camera_name][num_frames] = reproj_error

        elif calibrate_extrinsics_only:
            file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)

            # read intrinsics
            mtx = file_handle.getNode('CameraMatrix').mat()
            dist = file_handle.getNode('DistortionCoeffs').mat()
            file_handle.release()

            # calibrate extrinsics
            (rvec, tvec) = calibrate_extrinsics_from_image(camera_path + '/chess.jpg', mtx, dist)

            # save calibration
            file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_WRITE)
            file_handle.write('CameraMatrix', mtx)
            file_handle.write('DistortionCoeffs', dist)
            file_handle.write('Rotation', rvec)
            file_handle.write('Translation', tvec)
            file_handle.release()
        else:
            file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)

            # read intrinsics
            mtx = file_handle.getNode('CameraMatrix').mat()
            dist = file_handle.getNode('DistortionCoeffs').mat()

            # read extrinsics
            rvec = file_handle.getNode('Rotation').mat()
            tvec = file_handle.getNode('Translation').mat()

            file_handle.release()

        # ================================================================
        # visualize calibration
        # ================================================================
        
        # show axes
        if show_axes:
            image_path = camera_path + '/chess.jpg'
            frame = cv.imread(image_path)

            if frame is not None:
                # Draw and display the axes
                axes_3d = (5 * tile_size) * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,-1]], np.float32)
                axes_2d, jac = cv.projectPoints(axes_3d, rvec, tvec, mtx, dist)

                # Dibujar los ejes X (rojo), Y (verde), Z (azul)
                cv.line(frame, tuple(axes_2d[0][0].astype(int)), tuple(axes_2d[1][0].astype(int)), (0, 0, 255), 2)  # X
                cv.line(frame, tuple(axes_2d[0][0].astype(int)), tuple(axes_2d[2][0].astype(int)), (0, 255, 0), 2)  # Y
                cv.line(frame, tuple(axes_2d[0][0].astype(int)), tuple(axes_2d[3][0].astype(int)), (255, 0, 0), 2)  # Z

                cv.imshow('Checkerboard with 3D Axes', frame)
                cv.waitKey(0)
            cv.destroyAllWindows()

    if analysis_frames:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))

        for cam, data in reprojection_results.items():
            x = sorted(data.keys())
            y = [data[n] for n in x]
            plt.plot(x, y, marker='o', linestyle='-', label=cam)

        plt.xlabel("Number of Frames Used")
        plt.ylabel("Mean Reprojection Error (pixels)")
        plt.title("Reprojection Error vs Number of Frames (per Camera)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
     


if __name__ == "__main__":
    main()
