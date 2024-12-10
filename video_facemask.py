import cv2
import numpy as np
from keras._tf_keras.keras.models import load_model

# Load the saved model
model = load_model('model2.h5')

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start video capture
cap = cv2.VideoCapture(2)

def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (150, 150))
    face_array = np.expand_dims(face_img, axis=0) / 255.0
    return face_array

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for face detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4)

        for (x, y, w, h) in faces:
            # Extract face region
            face_img = frame[y:y + h, x:x + w]

            # Preprocess face image for prediction
            face_array = preprocess_face(face_img)

            # Make prediction
            prediction = model.predict(face_array)[0][0]
            label = "NO MASK lah" if prediction > 0.5 else "MASK"
            color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show the frame
        cv2.imshow('Live Face Mask Detector', frame)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
