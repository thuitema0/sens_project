import glm
import random
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment

# global variables
save_histograms = True
tracking = True

#tracking= True
block_size = 1.0
voxel_size = 45.0   # voxel every 4.5cm (30)
lookup_table = []
camera_handles = []
background_models = []
centers_A = []
centers_B = []
centers_C = []
centers_D = []
track_A=[]
track_B=[]
track_C=[]
track_D=[]

# generate the floor grid locations
def generate_grid(width, depth):
    data, colors = [], []
    for x in range(width):
        for z in range(depth):
            data.append([x * block_size - width / 2, -block_size, z * block_size - depth / 2])
            colors.append([1.0, 1.0, 1.0] if (x + z) % 2 == 0 else [0, 0, 0])
    return data, colors

# Take projections of voxels to a camera
def project_voxels_to_camera(voxels, width, height, depth, image):
    image_points = []       

    for i, coord in enumerate(voxels):
        x=coord[0]
        y=coord[2]
        z=coord[1]

        voxel_index = int(z + y * depth + x * (depth * height))

        i_camera = 1

        # Use lookup_table to find previously saved projections
        projection_x = int(lookup_table[i_camera][voxel_index][0][0])
        projection_y = int(lookup_table[i_camera][voxel_index][0][1])

        image_points.append([int(projection_y), int(projection_x)])
        
    return image_points

# Take only the upper body
def half_pixels_function(img_points):
    max_y = max(coord[0] for coord in img_points)
    min_y = min(coord[0] for coord in img_points)
    y_line = (max_y + min_y) / 2 + 20

    img_points_half=[]

    for coord in img_points:
        if coord[0] < y_line and coord[0]>(min_y+40): #supress lower part and head
            img_points_half.append(coord)

    return  img_points_half

# determines which voxels should be set
def set_voxel_positions(width, height, depth, curr_time,frame_cnt):
    global centers_A, centers_B, centers_C, centers_D, save_histograms, tracking, track_A, track_B, track_C, track_D

    if len(lookup_table) == 0:
        create_lookup_table(width, height, depth)

    # initialize voxel list
    voxel_list = []
    
    # swap y and z
    voxel_grid = np.ones((width, depth, height), np.float32)
    
    for i_camera in range(4):
        path_name = './data/cam' + str(i_camera + 1)

        if curr_time == 0:
            # train MOG2 on background video, remove shadows, default learning rate
            background_models.append(cv.createBackgroundSubtractorMOG2())
            background_models[i_camera].setShadowValue(0)

            # open background.avi
            camera_handle = cv.VideoCapture(path_name + '/background.avi')
            num_frames = int(camera_handle.get(cv.CAP_PROP_FRAME_COUNT))

            # train background model on each frame
            for i_frame in range(num_frames):
                ret, image = camera_handle.read()
                if ret:
                    background_models[i_camera].apply(image)

            # close background.avi
            camera_handle.release()

            # open video.avi
            camera_handles.append(cv.VideoCapture(path_name + '/video.avi'))
            num_frames = int(camera_handles[i_camera].get(cv.CAP_PROP_FRAME_COUNT))
 
        # read frame
        camera_handles[i_camera].set(cv.CAP_PROP_POS_FRAMES, frame_cnt)
        ret, image = camera_handles[i_camera].read()
        
        # determine foreground
        foreground_image = background_subtraction(image, background_models[i_camera])

        # set voxel to off if it is not visible in the camera, or is not in the foreground
        for x in range(width):
            for y in range(height):
                for z in range(depth):
                    if not voxel_grid[x, z, y]:
                        continue
                    voxel_index = z + y * depth + x * (depth * height)
                    try:
                        projection_x = int(lookup_table[i_camera][voxel_index][0][0])
                        projection_y = int(lookup_table[i_camera][voxel_index][0][1])
                        if projection_x < 0 or projection_y < 0 or projection_x >= foreground_image.shape[1] or projection_y >= foreground_image.shape[0] or not foreground_image[projection_y, projection_x]:
                            voxel_grid[x, z, y] = 0.0
                    except:
                        pass
   
    # Retrieve four voxels, corresponding to each person
    A, B, C, D, voxel_grid_clusters, center = get_voxel_clusters(voxel_list, voxel_grid)

    differences=[]

    # Get projections of each person in camera 2 and 3 (which gave us the best results)
    for i_camera_loop in range(1,3):
        colors = []
        cont=0

        # put voxels that are on in list
        # In this case it plots the voxels based on a new voxel grid where the clusters 
        # are differentiated by the value contained in the voxel (1,2,3 or 4).
        camera_handles[i_camera_loop].set(cv.CAP_PROP_POS_FRAMES, frame_cnt)
        ret, image = camera_handles[i_camera_loop].read()
        
        # Get projection of each person for the current camera
        img_point_A = project_voxels_to_camera(np.array(A), width, height, depth, image.shape)
        img_point_B = project_voxels_to_camera(np.array(B), width, height, depth, image.shape)
        img_point_C = project_voxels_to_camera(np.array(C), width, height, depth, image.shape)
        img_point_D = project_voxels_to_camera(np.array(D), width, height, depth, image.shape)

        img_point_A_half = half_pixels_function(img_point_A)
        img_point_B_half = half_pixels_function(img_point_B)
        img_point_C_half = half_pixels_function(img_point_C)
        img_point_D_half = half_pixels_function(img_point_D)
               
        image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

        # Save color histogram for each person for this camera
        HSV_A = get_color_histogram(image, img_point_A_half)
        HSV_B = get_color_histogram(image, img_point_B_half)
        HSV_C = get_color_histogram(image, img_point_C_half)
        HSV_D = get_color_histogram(image, img_point_D_half)
        
        # Save histograms if that option is turned on
        if save_histograms:
            np.save("histogram_HSV_A.npy", np.array(HSV_A,dtype=np.float64))
            np.save("histogram_HSV_B.npy", np.array(HSV_B,dtype=np.float64))
            np.save("histogram_HSV_C.npy", np.array(HSV_C,dtype=np.float64))
            np.save("histogram_HSV_D.npy", np.array(HSV_D,dtype=np.float64))
            save_histograms=False

        # Create arrays with histograms for each person.
        histograms = [
            HSV_A,
            HSV_B,
            HSV_C,
            HSV_D
        ]

        # Load old histograms
        histograms_old = [
            np.load("histogram_HSV_A.npy"),
            np.load("histogram_HSV_B.npy"),
            np.load("histogram_HSV_C.npy"),
            np.load("histogram_HSV_D.npy")
        ]

        # Convert to np.float32 datatype
        histograms = np.array(histograms, dtype=np.float32)
        histograms_old = np.array(histograms_old, dtype=np.float32)

        # Compare the difference between each histogram with each reference histogram
        histograms_differences = np.zeros((4, 4))

        # Calculate a 4 by 4 grid of histogram differences where we have the new histograms on one axis, and the old ones on the other
        for i in range(4):
            for j in range(4):
                histograms_differences[i, j] = get_histogram_diff(histograms_old[i], histograms[j])
        
        differences.append(histograms_differences)

    # Combine the histogram differences for all used cameras (in this case 2)
    summed_differences = differences[0]+differences[1]# +differences[2]+differences[3]
        
    # Run the linear_sum_assignment to use the hungarian algorithm to get the optimal result
    best_fit_row, best_fit_column = linear_sum_assignment(summed_differences)
    
    # Append the right center to the right person
    centers_A.append(center[best_fit_column[0]])
    centers_B.append(center[best_fit_column[1]])
    centers_C.append(center[best_fit_column[2]])
    centers_D.append(center[best_fit_column[3]])

    # Add the right label to the clusters used for the visualization
    clusters=[A, B, C, D]

    new_voxel_grid = np.zeros(voxel_grid.shape,dtype=np.float32)

    A_new=clusters[best_fit_column[0]]
    B_new= clusters[best_fit_column[1]]
    C_new=clusters[best_fit_column[2]]
    D_new=clusters[best_fit_column[3]]

    # Label all voxels in the new_voxel_grid with the corresponding person
    for voxel in A_new:
        x=voxel[0]
        z=voxel[1]
        y=voxel[2]
        new_voxel_grid[x][z][y] = 1.0

    for voxel in B_new:
        x=voxel[0]
        z=voxel[1]
        y=voxel[2]
        new_voxel_grid[x][z][y] = 2.0
    
    for voxel in C_new:
        x=voxel[0]
        z=voxel[1]
        y=voxel[2]
        new_voxel_grid[x][z][y] = 3.0
    
    for voxel in D_new:
        x=voxel[0]
        z=voxel[1]
        y=voxel[2]
        new_voxel_grid[x][z][y] = 4.0

    # Check the entire new_voxel_grid. If the voxel in the grid is labeled, add it to the voxel_list and append the corresponding color to colors.
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                if new_voxel_grid[x, z, y] == 1.0:
                    z1=y
                    y1=z
                    voxel_list.append([x * block_size - width / 2, z1 * block_size, y1 * block_size - depth / 2])
                    
                    colors.append([1.0,0,0])
                elif new_voxel_grid[x, z, y] == 2.0:
                    z1=y
                    y1=z
                    voxel_list.append([x * block_size - width / 2, z1 * block_size, y1 * block_size - depth / 2])
                    colors.append([0,1.0,0])
                elif new_voxel_grid[x, z, y] == 3.0:
                    z1=y
                    y1=z
                    voxel_list.append([x * block_size - width / 2, z1 * block_size, y1 * block_size - depth / 2])
                    colors.append([0,0,1.0])
                elif new_voxel_grid[x, z, y] == 4.0:
                    z1=y
                    y1=z
                    voxel_list.append([x * block_size - width / 2, z1 * block_size, y1 * block_size - depth / 2])
                    colors.append([1.0,1.0,0])

    # Place voxels at the old positions of a person
    if tracking == True:
        # If tracking is based on previous positions, skip steps where the difference in centers is too large.
        threshold = 30

        track_A=[]
        cont = 0
        count_skipped = 0

        for points_A in centers_A:
            # Always add the first point
            if cont == 0:
                track_A.append(points_A)
                cont += 1
            else:
                # Add the current point if the distance between this and the previous one is not too big, or if too many have been skipped. Reset the amount of skipped points
                pp=track_A[cont-1]

                if ((abs(points_A[0] - pp[0]) < threshold) and (abs(points_A[1] - pp[1]) < threshold)) or count_skipped > 10:
                    track_A.append(points_A)
                    cont += 1
                    count_skipped = 0
                else:
                    # Add one to the amount skipped
                    count_skipped += 1

        
        track_B=[]
        cont = 0
        count_skipped = 0
        for points_B in centers_B:
            if cont == 0:
                track_B.append(points_B)
                cont += 1
            else:
                pp=track_B[cont-1]
                if ((abs(points_B[0] - pp[0]) < threshold) and (abs(points_B[1] - pp[1]) < threshold)) or count_skipped > 10:
                    track_B.append(points_B)
                    cont += 1
                    count_skipped = 0
                else:
                    count_skipped += 1

        
        track_C=[]
        cont = 0
        count_skipped = 0
        for points_C in centers_C:
            if cont == 0:
                track_C.append(points_C)
                cont += 1
            else:
                pp=track_C[cont-1]
                if ((abs(points_C[0] - pp[0]) < threshold) and (abs(points_C[1] - pp[1]) < threshold)) or count_skipped > 10:
                    track_C.append(points_C)
                    cont += 1
                    count_skipped = 0
                else:
                    count_skipped += 1

        
        track_D=[]
        cont = 0
        count_skipped = 0
        for points_D in centers_D:
            if cont == 0:
                track_D.append(points_D)
                cont += 1
            else:
                pp=track_D[cont-1]
                if ((abs(points_D[0] - pp[0]) < threshold) and (abs(points_D[1] - pp[1]) < threshold)) or count_skipped > 10:
                    track_D.append(points_D)
                    cont += 1
                    count_skipped = 0
                else:
                    count_skipped += 1
        
        # Add all centers that were "allowed" by the tracking to the voxel list, with corresponding colours
        for points_A in track_A:
            y=points_A[1]
            x=points_A[0]
            z=0
            voxel_list.append([x * block_size - width / 2, z * block_size, y * block_size - depth / 2])
            colors.append([1.0,0,0])
        
        for points_A in track_B:
            y=points_A[1]
            x=points_A[0]
            z=0
            voxel_list.append([x * block_size - width / 2, z * block_size, y * block_size - depth / 2])
            colors.append([0,1.0,0])

        for points_A in track_C:
            y=points_A[1]
            x=points_A[0]
            z=0
            voxel_list.append([x * block_size - width / 2, z * block_size, y * block_size - depth / 2])
            colors.append([0,0,1.0])
        
        for points_A in track_D:
            y=points_A[1]
            x=points_A[0]
            z=0
            voxel_list.append([x * block_size - width / 2, z * block_size, y * block_size - depth / 2])
            colors.append([1.0,1.0,0])

    # Add all centers to the voxel_list without filtering 
    else:
        for points_A in centers_A:
            y=points_A[1]
            x=points_A[0]
            z=0
            voxel_list.append([x * block_size - width / 2, z * block_size, y * block_size - depth / 2])
            colors.append([1.0,0,0])
        
        for points_B in centers_B:
            y=points_B[1]
            x=points_B[0]
            z=0
            voxel_list.append([x * block_size - width / 2, z * block_size, y * block_size - depth / 2])
            colors.append([0,1.0,0])
        
        for points_C in centers_C:
            y=points_C[1]
            x=points_C[0]
            z=0
            voxel_list.append([x * block_size - width / 2, z * block_size, y * block_size - depth / 2])
            colors.append([0,0,1.0])
        
        for points_D in centers_D:
            y=points_D[1]
            x=points_D[0]
            z=0
            voxel_list.append([x * block_size - width / 2, z * block_size, y * block_size - depth / 2])
            colors.append([1.0,1.0,0])

    return voxel_list, colors

# get the color histogram of the selection
def get_color_histogram(image, pixel_list):
    colors = []
    for pixel in pixel_list:
        pixel = np.array(pixel)
        x = pixel[0]
        y = pixel[1]
        colors.append(image[x, y])
    
    colors = np.array(colors)
    
    # Seperate color channels
    colors_H = colors[:, 0]
    colors_S = colors[:, 1]
    colors_V = colors[:, 2]

    # Create histogram for each color channel
    histogram_H = cv.calcHist([colors_H], [0], None, [20], [0, 256]) / len(colors)
    histogram_S = cv.calcHist([colors_S], [0], None, [20], [0, 256]) / len(colors)
    histogram_V = cv.calcHist([colors_V], [0], None, [20], [0, 256]) / len(colors)

    return [histogram_H, histogram_S, histogram_V]

# Get the difference between two histograms
def get_histogram_diff(histogram_old, histogram_new):
    diff_total = 0

    # Compare the histograms for each channel, and create a total difference value
    for i in range(3):
        diff = cv.compareHist(histogram_old[i], histogram_new[i], cv.HISTCMP_BHATTACHARYYA)
        diff_total += diff
    
    return diff_total

# Get clusters of voxels by using k-means
def get_voxel_clusters(voxel_list, voxel_grid):
    # Initialize a 2d mapping of voxels
    voxels_2d = []

    # Add the maximum value for each column to the voxel_list. If one voxel in a column is on, this means that the corresponding value in the 2D-array voxel_list_2 is on.
    voxel_list_2 = np.max(np.asarray(voxel_grid), 2)
    voxel_list_2 = np.array(voxel_list_2, dtype=np.uint8)

    # Erode the areas that are "on"
    kernel = np.ones((5,5), np.uint8)
    im_result = cv.erode(voxel_list_2, kernel, iterations=1)  

    # Seperate all connected components
    blobs, im_with_separated_blobs, stats, _ = cv.connectedComponentsWithStats(im_result)

    sizes = stats[:, -1]
    sizes = sizes[1:]

    blobs -= 1

    # output image with only the kept components
    im_result = np.zeros_like(im_with_separated_blobs)
    im_result = im_result.astype(np.uint8)
    
    # for every component in the image, keep it only if it's above min_size
    for blob in range(blobs):
        if sizes[blob] >= 15:
            im_result[im_with_separated_blobs == blob + 1] = 255

    # Put all coordinates with a value of 255 in a list
    for i in range(im_result.shape[0]):
        for j in range(im_result.shape[1]):
            if im_result[i, j] == 255:
                voxels_2d.append([i, j])
    
    voxels_2d= np.array(voxels_2d, dtype=np.float32)
    im_result = im_result.astype(np.uint8)
    im_result = cv.cvtColor(im_result, cv.COLOR_GRAY2BGR)
    
    # define criteria and apply kmeans()
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 1.0) #10 before
    ret,label,center=cv.kmeans(voxels_2d,4,None,criteria,100,cv.KMEANS_RANDOM_CENTERS)
    
    # Seperate voxels into clusters
    A = np.array(voxels_2d[label.ravel()==0], dtype=np.uint32)
    B = np.array(voxels_2d[label.ravel()==1], dtype=np.uint32)
    C = np.array(voxels_2d[label.ravel()==2], dtype=np.uint32)
    D = np.array(voxels_2d[label.ravel()==3], dtype=np.uint32)

    # Go through each cluster and find the 3d coordinates that are "on" in that cluster
    A_3d = []
    
    # We create a new_voxel_grid in which we are going to save the value corresponding 
    #to the cluster in the corresponding value
    new_voxel_grid = np.zeros(voxel_grid.shape,dtype=np.float32)
    
    # Go through each 2D coordinate that belongs to cluster A, and check if any of the voxels in the corresponding column is "on". 
    # Label these voxels with the correct cluster in the new_voxel_grid
    for coord_2d in A:
        x = int(coord_2d[0])
        z = int(coord_2d[1])
        coords_3d = voxel_grid[x, z, :]
        cont=0
        for voxel  in coords_3d:
            cont=cont+1
            if voxel == 1.0:
                A_3d.append([x, z, cont])
                new_voxel_grid[x][z][cont] = 1.0

    # Do the same for cluster B, C, and D
    B_3d = []
    for coord_2d in B:
        x = int(coord_2d[0])
        z = int(coord_2d[1])
        coords_3d = voxel_grid[x, z, :]
        cont=0
        for voxel in coords_3d:
            cont=cont+1
            if voxel == 1.0:
                B_3d.append([x, z, cont])
                new_voxel_grid[x][z][cont] = 2.0

    C_3d = []
    for coord_2d in C:
        x = int(coord_2d[0])
        z = int(coord_2d[1])
        coords_3d = voxel_grid[x, z, :]
        cont=0
        for voxel in coords_3d:
            cont=cont+1
            if voxel == 1.0:
                C_3d.append([x, z, cont])
                new_voxel_grid[x][z][cont] = 3.0

    D_3d = []
    for coord_2d in D:
        x = int(coord_2d[0])
        z = int(coord_2d[1])
        coords_3d = voxel_grid[x, z, :]
        cont=0
        for voxel in coords_3d:
            cont=cont+1
            if voxel == 1.0:
                D_3d.append([x, z, cont])
                new_voxel_grid[x][z][cont] = 4.0

    return A_3d, B_3d, C_3d, D_3d, new_voxel_grid, center

# Create an image from the resulting paths
def create_path_image(height, width):
    global centers_A
    global centers_B
    global centers_C
    global centers_D

    global track_A
    global track_B
    global track_C
    global track_D 

    # Initialize the image for tracking, and for not tracking based on the previous position
    # Make the image larger so it can be viewed more easily.
    path_image = np.zeros((height * 4, width* 4, 3), np.uint8)
    path_image_tracking = np.zeros((height * 4, width* 4, 3), np.uint8)

    track_A = np.array(track_A, np.int32)
    track_B = np.array(track_B, np.int32)
    track_C = np.array(track_C, np.int32)
    track_D = np.array(track_D, np.int32)

    # Colour the correct pixels that are on the path with tracking
    for c in np.array(track_A, np.uint8):
        for i in range(4):
            for j in range(4):
                path_image_tracking[c[0] * 4 + i, c[1] * 4 + j] = [0, 0, 255]

    for c in np.array(track_B, np.uint8):
        for i in range(4):
            for j in range(4):
                path_image_tracking[c[0] * 4 + i, c[1] * 4 + j] = [0, 255, 0]
    
    for c in np.array(track_C, np.uint8):
        for i in range(4):
            for j in range(4):
                path_image_tracking[c[0] * 4 + i, c[1] * 4 + j] = [255, 0, 0]
    
    for c in np.array(track_D, np.uint8):
        for i in range(4):
            for j in range(4):
                path_image_tracking[c[0] * 4 + i, c[1] * 4 + j] = [0, 255, 255]

    # Plot the arrows between the steps
    # Disclaimer: blue and green should be swapped 
    for i in range(track_A.shape[0] - 1):
        plt.arrow(track_A[i, 0], track_A[i, 1], track_A[i + 1, 0] - track_A[i, 0], track_A[i + 1, 1] - track_A[i, 1], head_width=2, head_length=2.5, fc='r', ec='r')
    for i in range(track_B.shape[0] - 1):
        plt.arrow(track_B[i, 0], track_B[i, 1], track_B[i + 1, 0] - track_B[i, 0], track_B[i + 1, 1] - track_B[i, 1], head_width=2, head_length=2.5, fc='b', ec='b')
    for i in range(track_C.shape[0] - 1):
        plt.arrow(track_C[i, 0], track_C[i, 1], track_C[i + 1, 0] - track_C[i, 0], track_C[i + 1, 1] - track_C[i, 1], head_width=2, head_length=2.5, fc='g', ec='g')
    for i in range(track_D.shape[0] - 1):
        plt.arrow(track_D[i, 0], track_D[i, 1], track_D[i + 1, 0] - track_D[i, 0], track_D[i + 1, 1] - track_D[i, 1], head_width=2, head_length=2.5, fc='y', ec='y')
    
    plt.xlim(0, width)
    plt.ylim(0, height)

    plt.show()

    # Do the same for the image where no tracking based on the previous positions is applied
    centers_A = np.array(centers_A, np.int32)
    centers_B = np.array(centers_B, np.int32)
    centers_C = np.array(centers_C, np.int32)
    centers_D = np.array(centers_D, np.int32)

    for c in np.array(centers_A, np.uint8):
        for i in range(4):
            for j in range(4):

                path_image[c[0] * 4 + i, c[1] * 4 + j] = [0, 0, 255]

    for c in np.array(centers_B, np.uint8):
        for i in range(4):
            for j in range(4):
                path_image[c[0] * 4 + i, c[1] * 4 + j] = [255, 0, 0]
    
    for c in np.array(centers_C, np.uint8):
        for i in range(4):
            for j in range(4):
                path_image[c[0] * 4 + i, c[1] * 4 + j] = [0, 255, 0]
    
    for c in np.array(centers_D, np.uint8):
        for i in range(4):
            for j in range(4):
                path_image[c[0] * 4 + i, c[1] * 4 + j] = [0, 255, 255]

    for i in range(centers_A.shape[0] - 1):
        plt.arrow(centers_A[i, 0], centers_A[i, 1], centers_A[i + 1, 0] - centers_A[i, 0], centers_A[i + 1, 1] - centers_A[i, 1], head_width=2, head_length=2.5, fc='r', ec='r')
    for i in range(centers_B.shape[0] - 1):
        plt.arrow(centers_B[i, 0], centers_B[i, 1], centers_B[i + 1, 0] - centers_B[i, 0], centers_B[i + 1, 1] - centers_B[i, 1], head_width=2, head_length=2.5, fc='b', ec='b')
    for i in range(centers_C.shape[0] - 1):
        plt.arrow(centers_C[i, 0], centers_C[i, 1], centers_C[i + 1, 0] - centers_C[i, 0], centers_C[i + 1, 1] - centers_C[i, 1], head_width=2, head_length=2.5, fc='g', ec='g')
    for i in range(centers_D.shape[0] - 1):
        plt.arrow(centers_D[i, 0], centers_D[i, 1], centers_D[i + 1, 0] - centers_D[i, 0], centers_D[i + 1, 1] - centers_D[i, 1], head_width=2, head_length=2.5, fc='y', ec='y')
    
    plt.xlim(0, width)
    plt.ylim(0, height)

    plt.show()

    cv.imshow("path_image", path_image)
    cv.waitKey(0)
    cv.imwrite("center_paths.png", path_image)
    cv.imwrite("center_paths_tracking.png", path_image_tracking)


# create lookup table
def create_lookup_table(width, height, depth):
    # create 3d voxel grid
    voxel_space_3d = []
    for x in range(width):
        for y in range(height):
            for z in range(depth):
                voxel_space_3d.append([voxel_size * (x * block_size - width / 2), voxel_size * (z * block_size - depth / 2), - voxel_size * (y * block_size)])

    for i_camera in range(4):
        camera_path = './data/cam' + str(i_camera + 1)

        # use config.xml to read camera calibration
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        mtx = file_handle.getNode('CameraMatrix').mat()
        dist = file_handle.getNode('DistortionCoeffs').mat()
        rvec = file_handle.getNode('Rotation').mat()
        tvec = file_handle.getNode('Translation').mat()
        file_handle.release()
        
        # project voxel 3d points to 2d in each camera
        voxel_space_2d, jac = cv.projectPoints(np.array(voxel_space_3d, np.float32), rvec, tvec, mtx, dist)
        lookup_table.append(voxel_space_2d)


# applies background subtraction to obtain foreground mask
def background_subtraction(image, background_model):
    foreground_image = background_model.apply(image, learningRate=0)
    
    # remove noise through dilation and erosion
    erosion_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    dilation_elt = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    foreground_image = cv.dilate(foreground_image, dilation_elt)
    foreground_image = cv.erode(foreground_image, erosion_elt)
            
    return foreground_image


# Gets stored camera positions
def get_cam_positions():
    cam_positions = []
    
    for i_camera in range(4):
        camera_path = './data/cam' + str(i_camera + 1)
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        tvec = file_handle.getNode('Translation').mat()
        rvec = file_handle.getNode('Rotation').mat()
        file_handle.release()
        # obtain positions
        rotation_matrix = cv.Rodrigues(rvec)[0]
        positions = -np.matrix(rotation_matrix).T * np.matrix(tvec / voxel_size)
        cam_positions.append([positions[0][0], -positions[2][0], positions[1][0]])
    return cam_positions, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0], [1.0, 1.0, 0]]


# Gets stored camera rotations
def get_cam_rotation_matrices():
    cam_rotations = []
    
    for i in range(4):
        camera_path = './data/cam' + str(i + 1)
        file_handle = cv.FileStorage(camera_path + '/config.xml', cv.FileStorage_READ)
        rvec = file_handle.getNode('Rotation').mat()
        file_handle.release()

        # # normalize rotations
        angle = np.linalg.norm(rvec)
        axis = rvec / angle

        # apply rotation to compensate for difference between OpenCV and OpenGL
        transform = glm.rotate(-0.5 * np.pi, [0, 0, 1]) * glm.rotate(-angle, glm.vec3(axis[0][0], axis[1][0], axis[2][0]))
        transform_to = glm.rotate(0.5 * np.pi, [1, 0, 0])
        transform_from = glm.rotate(-0.5 * np.pi, [1, 0, 0])
        cam_rotations.append(transform_to * transform * transform_from)
    return cam_rotations