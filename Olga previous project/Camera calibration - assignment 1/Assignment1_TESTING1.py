import cv2 as cv
import numpy as np
import glob


#Load camera matrix and distortion coefficients from our training data
cameraMatrix = np.load(r'cameraMatrix.pkl', allow_pickle=True)
dist = np.load(r'dist.pkl', allow_pickle=True)

#CUBE+PROJECTION DRAWING

#Define termination criteria for subpixel corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#Define object points for chessboard corners
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 2  # Multiply by 2 for 2 cm grid

#Define cube points in 3D space with adjusted dimensions
cube_points_3d = np.float32([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0],
                             [0, 0, -4], [4, 0, -4], [4, 4, -4], [0, 4, -4]])

#Define the axis points for drawing the 3D axes
axis = np.float32([[5,0,0], [0,5,0], [0,0,-5]])

#Function to draw axes (not final)
def draw_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[0].ravel())), (51, 51, 51), 5)  # X-axis (Blue)
    img = cv.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[1].ravel())), (51, 51, 51), 5)  # Y-axis (Green)
    img = cv.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[2].ravel())), (51, 51, 51), 5)  # Z-axis (Red)
    return img

#Define function to draw cube
def draw_cube(img, corners, rvec, tvec):
    imgpts, _ = cv.projectPoints(cube_points_3d, rvec, tvec, cameraMatrix, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    #Draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    
    #Draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
        
    #Draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img


#Start Webcam and press 's' to capture frame that you want to process with
def capture_video():
    cap = cv.VideoCapture(0)
    
    #Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        ret, img = cap.read()
        img = cv.resize(img, None, fx=0.5, fy=0.5, interpolation=cv.INTER_AREA)
        cv.imshow('Input', img)

        c = cv.waitKey(1)
        if c == 27:
            break
        elif c == ord('s'):  # Check if the "s" key is pressed
            #Capture the current frame
            ret, img = cap.read()
            #Save the captured frame as a .jpg file
            cv.imwrite('captured_frame.jpg', img)
            print("Frame saved as captured_frame.jpg")
            #Return the captured frame
            return img

    cap.release()
    cv.destroyAllWindows()
    
    
#Check if the user wants to use webcam or provide an image
use_webcam = input("Do you want to use webcam? (yes/no): ")
if use_webcam.lower() == 'yes':
    
    #Call the function and store the returned frame
    img = capture_video()

    #Convert the captured frame to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #Define the chessboard size
    chessboardSize = (6, 9)

    #Find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    #If corners are found, refine them and draw them on the image
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        
        #Solve PnP to find rotation and translation vectors
        ret, rvec, tvec = cv.solvePnP(objp, corners2, cameraMatrix, dist)
        
        #Project 3D points to image plane (not final)
        imgpts, _ = cv.projectPoints(axis, rvec, tvec, cameraMatrix, dist)
        
        #Draw world 3D axes (not final)
        img = draw_axes(img, corners2, imgpts)
        
        #Draw cube on the image
        img = draw_cube(img, corners2, rvec, tvec)


    #Display the image with chessboard corners, axes, and cube
    cv.imshow('Captured Frame', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    
else:
   test_image_dir = r'Test images'
   test_images = glob.glob(test_image_dir + r'\*.png')
   print("Test images found:", test_images)
   
   for test_image_path in test_images:
        test_image = cv.imread(test_image_path)
        if test_image is not None:
            
            #Convert the captured frame to grayscale
            gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)

            #Define the chessboard size
            chessboardSize = (6, 9)

            #Find chessboard corners
            ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

            #If corners are found, refine them and draw them on the image
            if ret==True:
                corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                cv.drawChessboardCorners(test_image, chessboardSize, corners, ret)
                
                          
                #Show the image in the window with the specified dimensions
                cv.namedWindow('Image', cv.WINDOW_NORMAL)
                cv.resizeWindow('Image', 800, 600)
                cv.imshow('Image', test_image)
                  
                #Solve PnP to find rotation and translation vectors
                ret, rvec, tvec = cv.solvePnP(objp, corners, cameraMatrix, dist)
                
                #Project 3D points to image plane
                imgpts, _ = cv.projectPoints(axis, rvec, tvec, cameraMatrix, dist)
                
                #Draw world 3D axes
                test_image = draw_axes(test_image, corners, imgpts)
                
                #Draw cube on the image
                test_image = draw_cube(test_image, corners, rvec, tvec)

            #Display the image with chessboard corners, axes, and cube
            cv.imshow('Captured Frame', test_image)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Invalid image path or unable to read the image.")
