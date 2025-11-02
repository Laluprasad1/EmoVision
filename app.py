import cv2
import mediapipe as mp
from deepface import DeepFace
import os

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=5, refine_landmarks=True, min_detection_confidence=0.5)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Webcam not detected.")
    exit()
else:
    print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape

            # Draw facial landmarks (eyes, eyebrows, lips)
            for lm in face_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            # Extract bounding box of face
            xs = [int(lm.x * w) for lm in face_landmarks.landmark]
            ys = [int(lm.y * h) for lm in face_landmarks.landmark]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            face_roi = frame[y1:y2, x1:x2]

            try:
                result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                emotions = result[0]['emotion']
                dominant = result[0]['dominant_emotion']

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Dominant: {dominant}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                y_offset = 20
                for emo, score in emotions.items():
                    cv2.putText(frame, f"{emo}: {score:.2f}%", (x1, y2 + y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    y_offset += 18

            except Exception as e:
                print("Emotion error:", e)

    cv2.imshow("Advanced Emotion Recognition (Facial Patterns)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
