import cv2
import os

# Output directory
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# Open both cameras
cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if not cam1.isOpened() or not cam2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

frame_idx = 0

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    if not ret1 or not ret2:
        print("Error: Failed to grab frames.")
        break

    # Show the live feeds
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        # Save both frames
        fname1 = os.path.join(out_dir, f"frame_{frame_idx:04d}_left.png")
        fname2 = os.path.join(out_dir, f"frame_{frame_idx:04d}_right.png")
        cv2.imwrite(fname1, frame1)
        cv2.imwrite(fname2, frame2)
        print(f"Saved {fname1} and {fname2}")
        frame_idx += 1

    elif key == 27:  # ESC key
        print("Exiting.")
        break

# Cleanup
cam1.release()
cam2.release()
cv2.destroyAllWindows()
