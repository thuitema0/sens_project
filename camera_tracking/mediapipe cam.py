import cv2
import mediapipe as mp

# Inicializa Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Dictionary with the landmark number correspondance to mediapipe
landmark_names = {
    33: "Left exocanthion",
    133: "Left endocanthion",
    6: "Nasion",
    362: "Right endocanthion",
    263: "Right exocanthion",
    4: "Pronasale",
    98: "Left alar crest",
    327: "Right alar crest",
    2: "Subnasale",
    61: "Left cheilion",
    291: "Right cheilion",
    0: "Labiale superius (outer)",
    17: "Labiale inferius (outer)",
    199: "Pogonion"
}

#To initialise the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #To convert to Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for facial_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(facial_landmarks.landmark):
                if idx in landmark_names:
                    h, w, _ = frame.shape
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 2, (0, 0, 0), -1)
                    cv2.putText(frame, landmark_names[idx], (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.2, (255, 0, 0), 1)

    cv2.imshow("Facial Landmarks - Anatomical Points", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to skip
        break

cap.release()
cv2.destroyAllWindows()
