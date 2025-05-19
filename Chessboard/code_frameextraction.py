import cv2

# Path to your .avi file
avi_file_path = r'./data_trial/cam2/intrinsics.mp4'

# Open the video file
video_capture = cv2.VideoCapture(avi_file_path)
# Check if the video file opened successfully
if not video_capture.isOpened():
    print("Error: Couldn't open the video file.")
    exit()

# Counter to keep track of frames
frame_count = 0

# Loop through the video frames
while True:
    # Read the next frame
    ret, frame = video_capture.read()

    # If frame is not read correctly or end of file, break the loop
    if not ret:
        break

    # Increment frame count
    frame_count += 1
     
    # Do something with the frame
    # For example, you can save it
    frame_output_path = f'frame_{frame_count}.jpg'
    cv2.imwrite(frame_output_path, frame)
    print(f"Frame {frame_count} saved as {frame_output_path}")

    # Display the frame (optional)
    cv2.imshow('Frame', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
video_capture.release()
cv2.destroyAllWindows()
