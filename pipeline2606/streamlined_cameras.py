import cv2
import os
import time

# Directory to save videos
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
save_dir = SCRIPT_DIR
os.makedirs(save_dir, exist_ok=True)

# Configuration for camera parameters
def configure_camera(cap, focus=100, exposure=-7, gain=0):
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_FOCUS, focus)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual exposure
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    cap.set(cv2.CAP_PROP_GAIN, gain)

# Generic recorder for a single camera
def record_single_camera(cam_index, output_name, window_title, message):
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    configure_camera(cap)

    # Get camera properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    size   = (width, height)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(save_dir, output_name + '.mp4'), fourcc, fps, size)

    recording = False
    print(message)

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read from camera {cam_index}")
            break

        # Overlay instructions/recording status
        display = frame.copy()
        if not recording:
            text = message
        else:
            text = f"Recording {output_name}, Press 'q' to stop"
        cv2.putText(display, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow(window_title, display)
        key = cv2.waitKey(1) & 0xFF

        # Start/stop recording on 'q'
        if key == ord('q'):
            if not recording:
                print(f"Started recording {output_name}.mp4")
                recording = True
            else:
                print(f"Stopped recording {output_name}.mp4")
                break

        # Write frame if recording
        if recording:
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyWindow(window_title)

# Recorder for two cameras simultaneously
def record_dual_cameras(cam1_idx, cam2_idx, out1_name, out2_name, window1, window2, message):
    cap1 = cv2.VideoCapture(cam1_idx, cv2.CAP_DSHOW)
    cap2 = cv2.VideoCapture(cam2_idx, cv2.CAP_DSHOW)
    configure_camera(cap1)
    configure_camera(cap2)

    w1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    h1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps1 = cap1.get(cv2.CAP_PROP_FPS) or 30
    size1 = (w1, h1)

    w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps2 = cap2.get(cv2.CAP_PROP_FPS) or 30
    size2 = (w2, h2)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out1 = cv2.VideoWriter(os.path.join(save_dir, out1_name + '.mp4'), fourcc, fps1, size1)
    out2 = cv2.VideoWriter(os.path.join(save_dir, out2_name + '.mp4'), fourcc, fps2, size2)

    recording = False
    print(message)

    while True:
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            print("Failed to read from one or both cameras")
            break

        # Overlay status on both
        disp1 = frame1.copy()
        disp2 = frame2.copy()
        if not recording:
            text = message
        else:
            text = f"Recording extrinsics, Press 'q' to stop"
        cv2.putText(disp1, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(disp2, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.imshow(window1, disp1)
        cv2.imshow(window2, disp2)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            if not recording:
                print("Started extrinsics recording")
                recording = True
            else:
                print("Stopped extrinsics recording")
                break

        if recording:
            out1.write(frame1)
            out2.write(frame2)

    cap1.release()
    cap2.release()
    out1.release()
    out2.release()
    cv2.destroyWindow(window1)
    cv2.destroyWindow(window2)

if __name__ == '__main__':
    # Record intrinsics for camera 1
    record_single_camera(
        cam_index=1,
        output_name='cam1_intrinsics',
        window_title='Camera 1 - Intrinsics',
        message="Press 'q' to start intrinsics recording for Cam1"
    )

    # Record intrinsics for camera 2
    record_single_camera(
        cam_index=2,
        output_name='cam2_intrinsics',
        window_title='Camera 2 - Intrinsics',
        message="Press 'q' to start intrinsics recording for Cam2"
    )

    # Record extrinsics with both cameras
    record_dual_cameras(
        cam1_idx=1,
        cam2_idx=2,
        out1_name='cam1_extrinsics',
        out2_name='cam2_extrinsics',
        window1='Camera 1 - Extrinsics',
        window2='Camera 2 - Extrinsics',
        message="Press 'q' to start extrinsics recording for both cameras"
    )
