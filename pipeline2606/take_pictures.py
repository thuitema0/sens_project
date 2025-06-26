import cv2
import os

# Configuration for camera parameters
def configure_camera(cap, focus=100, exposure=-7, gain=0):
    # Turn off autofocus, set focus
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, focus)
    # Switch to manual exposure and set it
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual mode on many backends
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    # Set gain
    cap.set(cv2.CAP_PROP_GAIN, gain)

# === MAIN ===

# Create output folder
out_dir = "output"
os.makedirs(out_dir, exist_ok=True)

# Open both cameras
cam1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(2, cv2.CAP_DSHOW)
if not cam1.isOpened() or not cam2.isOpened():
    print("Error: Could not open one or both cameras.")
    exit()

# Apply your custom configs
configure_camera(cam1, focus=150, exposure=-5, gain=0)
configure_camera(cam2, focus=150, exposure=-5, gain=0)

frame_idx = 0

while True:
    ret1, frame1 = cam1.read()
    ret2, frame2 = cam2.read()
    if not ret1 or not ret2:
        print("Error: Failed to grab frames.")
        break

    # Show live feeds
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        # Save left and right frames
        fname1 = os.path.join(out_dir, f"frame_{frame_idx:04d}_left.png")
        fname2 = os.path.join(out_dir, f"frame_{frame_idx:04d}_right.png")
        cv2.imwrite(fname1, frame1)
        cv2.imwrite(fname2, frame2)
        print(f"Saved {fname1} and {fname2}")
        frame_idx += 1
    elif key == 27:  # ESC
        print("Exiting.")
        break

# Cleanup
cam1.release()
cam2.release()
cv2.destroyAllWindows()
