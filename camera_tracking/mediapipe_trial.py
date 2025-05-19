import cv2 
import mediapipe as mp

#Face mesh
mp_face_mesh= mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()


#image = cv2.imread('person.jpg')
image = cv2.imread('person.jpg')
height, width, _ = image.shape
print("Height, width", height, width)

# Diccionario: índice de Mediapipe -> nombre anatómico
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


#Facial landmarks
result = face_mesh.process(image)
# for facial_landmarks in result.multi_face_landmarks:
#     for idx, lm in enumerate(facial_landmarks.landmark):
#         if idx in landmark_names:
#             h, w, _ = image.shape
#             x, y = int(lm.x * w), int(lm.y * h)
#             cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
#             #cv2.putText(image, landmark_names[idx], (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)


if result.multi_face_landmarks:
    for facial_landmarks in result.multi_face_landmarks:
        for idx, lm in enumerate(facial_landmarks.landmark):
            h, w, _ = image.shape
            x, y = int(lm.x * w), int(lm.y * h)
            print('Landmark:', idx, "  2D Coordinates: ", x, " ", y )
            cv2.circle(image, (x, y), 1, (0, 0, 0), -1)  # smaller green dot for clarity


cv2.imshow("Image", image)
cv2.waitKey(0)