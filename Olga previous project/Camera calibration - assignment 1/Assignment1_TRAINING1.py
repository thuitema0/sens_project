#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2
import cv2 as cv
import glob
import pickle

#Function to select manually the coordinates 
def click_event(event, x, y, flags, params): 

    #Checking for left mouse clicks 
    if event == cv.EVENT_LBUTTONDOWN: 
                       
        #Create the window with the specified dimensions and show the coordinates selected
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.resizeWindow('image', 1200, 600) #x_window, y_window
        cv.imshow('image', img)
        
        #Save the coordinates selected
        coordinates = []
        x=np.float32(x)
        y=np.float32(y)
        coordinates.append(x)
        coordinates.append(y)
        print(coordinates)
        corners.append(coordinates)
        
    #Checking for right mouse clicks      
    if event==cv.EVENT_RBUTTONDOWN: 
        
        #Create the window with the specified dimensions and show the coordinates selected
        cv.circle(img, (x, y), 5, (0, 255, 0), -1)
        cv.namedWindow('image', cv.WINDOW_NORMAL)
        cv.resizeWindow('image', 1200, 600)
        cv.imshow('image', img)
        
        #Save the coordinates selected
        coordinates = []
        x=np.float32(x)
        y=np.float32(y)
        coordinates.append(x)
        coordinates.append(y)
        print(coordinates)
        corners.append(coordinates)
        
def pespectivetransformation (width, height,img, corners_np):
    #Define the target points for interpolation
    dst_points = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype="float32")
    
    #Calculate the inverse transformation matrix
    matrix_inv = cv2.getPerspectiveTransform(corners_np, dst_points)
    
    #Apply the inverse transformation to the image
    img_transformed = cv2.warpPerspective(img, matrix_inv, (width, height))
    
    return img_transformed, matrix_inv
    
        
def interpolation_step (rows, cols, width, height):
    top_left = [0, 0]
    bottom_left = [0, height]
    top_right = [width, 0]
    
    #Interpolate points between top_left and top_right for rows
    x_values_top = np.linspace(top_left[0], top_right[0], cols)
    
    #Interpolate points between top_left and bottom_left for columns
    y_values_left = np.linspace(top_left[1], bottom_left[1], rows)
    
    #Use meshgrid to generate a matrix containing all combinations of interpolated points
    x_mesh, y_mesh = np.meshgrid(x_values_top, y_values_left)
    
    #Convert the meshgrid points to a flat array of coordinates
    result_points = np.vstack([x_mesh.flatten(), y_mesh.flatten()]).T
    
    return result_points
    
def inverse_transformation (all_points,matrix_inv):
    #Apply the inverse transformation matrix to map the points from the transformed image back to the original image
    original_points = cv2.perspectiveTransform(all_points.reshape(-1, 1, 2), np.linalg.inv(matrix_inv))
    
    #Convert the points to the type of interest
    original_points = original_points.reshape(-1, 2).astype(np.float32)
    
    return original_points

#Function to draw axes (not final)
def draw_axes(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[0].ravel())), (255, 0, 0), 5)  # X-axis (Blue)
    img = cv.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[1].ravel())), (0, 255, 0), 5)  # Y-axis (Green)
    img = cv.line(img, tuple(map(int, corner)), tuple(map(int, imgpts[2].ravel())), (0, 0, 255), 5)  # Z-axis (Red)
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

#STEP 1:  FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS

chessboardSize = (6,9) #We need to specify the size: (widht corners and height corners that we want to find (interior ones))
frameSize = (1680,2520) #diemnsions of the image

#termination criteria: Criteria for termination of the iterative search algorithm. It determines the maximum number of iterations and/or the required accuracy.
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

#prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20 
objp = objp * size_of_chessboard_squares_mm
def enhance_image(img):
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Apply histogram equalization to enhance contrast
    enhanced_img = cv.equalizeHist(gray)
    
    # Apply morphological gradient to enhance edges
    kernel = np.ones((5,5),np.uint8)
    gradient_img = cv.morphologyEx(enhanced_img, cv.MORPH_GRADIENT, kernel)
    
    return gradient_img

#Arrays to store object points and image points from all the images.
objpoints = [] #3d point in real world space
imgpoints = [] #2d points in image plane.


images = glob.glob(r'Cheesboard photos\*.jpg') #in order to open all the images from the folder

for image in images:
    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #we convert it to a greyscale image
    #Find the chess board corners using the function that opencv provides to find the corners of the chessboard.
    #input corresponds to input image, patternsize and None represents the optional output parameter corners
    #(pointer to the detected corners)--> By using None we are indicating that we are not interested in storing the detected corners?
    
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    #If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        #used to refine the positions of corners detected in an image. It is typically used after an initial corner detection algorithm
        #refines the positions of corners by iteratively searching for the minimum of a certain criterion (typically the difference in pixel intensity) within a small window around each corner.
        imgpoints.append(corners)

        #Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret) #now we put the color image, the refined corners.
        
        #Show the image in the window with the specified dimensions
        x_window = 800
        y_window = 600
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.resizeWindow('Image', x_window, y_window)
        cv.imshow('Image', img)
        cv.waitKey(1000)
        
    else:
        
        #Manual detection of outer corners and interpolation
        objpoints.append(objp)
        corners=[]
        manualdetection=True
        
        while manualdetection==True:
            enhanced_img = enhance_image(img)
            gray = enhanced_img
            
            #Show the image in the window with the specified dimensions
            x_window = 1200
            y_window = 600
            cv.namedWindow('image', cv.WINDOW_NORMAL)
            cv.resizeWindow('image', x_window, y_window)
            cv.imshow('image', img)
            #setting mouse handler for the image and calling the click_event() function 
            cv.setMouseCallback("image", click_event) 
            cv.waitKey(0) 
            cv.destroyAllWindows()
            
            #Check if the corners selected manually are correct:
            len_corners= len(corners)
            if len_corners == 4:
                manualdetection=False
            elif len_corners <4:
                print("Some corner is missing! Please Select it")
            elif len_corners > 4:
                print("Select another time the corners please! You have select more corners!")
                corners=[]
                
        corners = np.float32(corners)
        
        #Assessing the limits of the transformed image
        height, width = 600,900
        
        #Find transformation to the original image to create a transformed image
        img_transformed, matrix_inv=pespectivetransformation(width, height,img, corners)
        
        #Interpolation of the 4 outer corners in the transformed image to find all the corners of the chessboard
        all_points = interpolation_step (chessboardSize[0], chessboardSize[1], width, height)
        
        #Find the corners found in the original image
        original_points_flat = inverse_transformation (all_points, matrix_inv)
  
        
        corners = np.array(original_points_flat)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        #used to refine the positions of corners detected in an image. It is typically used after an initial corner detection algorithm
        #refines the positions of corners by iteratively searching for the minimum of a certain criterion (typically the difference in pixel intensity) within a small window around each corner.
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret) #now we put the color image, the refined corners.
        
        #Show the image in the window with the specified dimensions
        x_window = 800
        y_window = 600
        cv.namedWindow('Image', cv.WINDOW_NORMAL)
        cv.resizeWindow('Image', x_window, y_window)
        cv.imshow('Image', img)
        cv.waitKey(1000)
        
        corners2= corners2.reshape(54,1,2)
        imgpoints.append(corners2)

cv.destroyAllWindows()

#CALIBRATION STEP:

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)

#Save the camera calibration result for later use 
pickle.dump((cameraMatrix, dist), open( "calibration.pkl", "wb" ))
pickle.dump(cameraMatrix, open( "cameraMatrix.pkl", "wb" ))
pickle.dump(dist, open( "dist.pkl", "wb" ))


#CALCULATION OF REPROJECTION ERROR:

mean_error = 0

for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], cameraMatrix, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
    mean_error += error

print( "total error: {}".format(mean_error/len(objpoints)) )


#3D PLOT: with the locations of the camera relative to the chessboard when each of the training images was taken

camera_positions = []
for rvec, tvec, objp in zip(rvecs, tvecs, objpoints):
    #Convert rotation vector to rotation matrix
    R, _ = cv.Rodrigues(rvec)
    #Camera position is the negative of the transpose of the rotation matrix times the translation vector
    #This transformation brings points from world coordinates to camera coordinates
    camera_position = -R.T.dot(tvec.reshape(-1))
    camera_positions.append(camera_position)

camera_positions = np.array(camera_positions)

#Plot camera positions

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(-camera_positions[:, 2], camera_positions[:, 1], camera_positions[:, 0], c='r', marker='o')

# Plot the chessboard corners with x and z swapped
for objp in objpoints:
    ax.scatter(-objp[:, 2], objp[:, 1], objp[:, 0], c='b', marker='x')

ax.set_xlabel('Z')  
ax.set_ylabel('Y')
ax.set_zlabel('X')  
plt.show()

#CUBE+PROJECTION DRAWING

#Define termination criteria for subpixel corner refinement
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Define object points for chessboard corners
objp = np.zeros((9*6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 2  # Multiply by 2 for 2 cm grid

# Define cube points in 3D space with adjusted dimensions
cube_points_3d = np.float32([[0, 0, 0], [4, 0, 0], [4, 4, 0], [0, 4, 0],
                             [0, 0, -4], [4, 0, -4], [4, 4, -4], [0, 4, -4]])

# Define the axis points for drawing the 3D axes
axis = np.float32([[5,0,0], [0,5,0], [0,0,-5]])

# Process each image to draw a cube
for fname in glob.glob(r'\Cheesboard photos\*.jpg'):
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find chessboard corners
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)  # 6x9 grid
    if ret == True:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # Solve PnP to find rotation and translation vectors
        ret, rvec, tvec = cv.solvePnP(objp, corners2, cameraMatrix, dist)
        
        # Project 3D points to image plane (not final)
        imgpts, _ = cv.projectPoints(axis, rvec, tvec, cameraMatrix, dist)
        # Draw world 3D axes (not final)
        img = draw_axes(img, corners2, imgpts)
        
        # Draw cube on the image
        img = draw_cube(img, corners2, rvec, tvec)

        #Show the image in the window with the specified dimensions
        cv.namedWindow('img', cv.WINDOW_NORMAL)
        cv.resizeWindow('img', 800, 600)
        cv.imshow('img', img)

        # Wait for user input
        k = cv.waitKey(0) & 0xFF
        #if user presses 's', the image saves as .png
        if k == ord('s'):
            cv.imwrite(fname[:6] + '_cube.png', img)

cv.destroyAllWindows()