import cv2
import os

# === Parameters ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
#video_path = os.path.join(SCRIPT_DIR, "cam2_intrinsics.mp4")
frame_interval = 30

video_paths = [os.path.join(SCRIPT_DIR, "cam1/intrinsics/cam1_intrinsics.mp4"),
               os.path.join(SCRIPT_DIR, "cam1/extrinsics/cam1_extrinsics.mp4"),
               os.path.join(SCRIPT_DIR, "cam2/intrinsics/cam2_intrinsics.mp4"),
               os.path.join(SCRIPT_DIR, "cam2/extrinsics/cam2_extrinsics.mp4")
               ]

for video_path in video_paths:
    # === Set output directory ===
    video_dir = os.path.dirname(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    #output_dir = os.path.join(video_dir, f"{video_name}_frames")
    output_dir = video_dir

    # === Open video ===
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames to: {output_dir}")
  
