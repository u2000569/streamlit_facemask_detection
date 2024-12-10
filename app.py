import streamlit as st
import cv2
import os
import numpy as np
from keras._tf_keras.keras.models import load_model
from keras._tf_keras.keras.utils import img_to_array
from PIL import Image
from mtcnn import MTCNN

model = load_model('model2.h5')

# Function to predict mask status
def detect_and_predict_mask(image):
    import numpy as np
    import cv2
    from mtcnn import MTCNN

    # Convert PIL image to NumPy array
    annotated_image = np.array(image)

    # Ensure the image is in RGB format (if needed)
    if annotated_image.shape[-1] != 3:
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGBA2RGB)

    detector = MTCNN()
    faces = detector.detect_faces(annotated_image)

    mask_count, no_mask_count = 0, 0

    for face in faces:
        x, y, w, h = face['box']

        # Extract the face region
        face_img = annotated_image[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (150, 150))
        face_array = np.expand_dims(face_img, axis=0) / 255.0

        # Predict mask status
        prediction = model.predict(face_array)[0][0]
        if prediction > 0.5:
            label = "No Mask"
            color = (255, 0, 0)  # Red for "No Mask"
            no_mask_count += 1
        else:
            label = "Mask"
            color = (0, 255, 0)  # Green for "Mask"
            mask_count += 1

        # Annotate image
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), color, 2)
        cv2.putText(annotated_image, f"{label}", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return annotated_image, mask_count, no_mask_count



def live_webcam(camera_index):
    stframe = st.empty()  # Placeholder for the webcam stream
    cap = cv2.VideoCapture(camera_index)  # Open the selected camera

    if not cap.isOpened():
        st.error(f"Unable to access camera {camera_index}. Please select another camera.")
        return

    detector = MTCNN()

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to grab frame from the camera.")
            break

        # Flip and process frame
        frame = cv2.flip(frame, 1)  # Flip horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb_frame)

        for face in faces:
            x, y, w, h = face['box']
            face_img = rgb_frame[y:y+h, x:x+w]

            if face_img.size > 0:
                face_img = cv2.resize(face_img, (150, 150))
                face_array = np.expand_dims(face_img, axis=0) / 255.0
                prediction = model.predict(face_array)[0][0]
                label = "No Mask" if prediction > 0.5 else "Mask"
                color = (0, 0, 255) if prediction > 0.5 else (0, 255, 0)

                # Annotate frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} ({prediction:.2f})", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display the frame in Streamlit
        stframe.image(frame, channels="BGR", use_column_width=True)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

            
# Streamlit app
st.title("Face Mask Detector")
st.write("Upload an image or use the live webcam to detect masks.")

# Tabs for image upload and webcam
tabs = st.tabs(["Image Upload", "Live Webcam"])

# Tab for image upload
with tabs[0]:
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Detecting...")

        # Detect faces and predict mask status
        annotated_image, mask_count, no_mask_count = detect_and_predict_mask(image)

        # Display the annotated image and results
        st.image(annotated_image, caption='Processed Image', use_column_width=True)
        st.write(f"**Mask Count:** {mask_count}")
        st.write(f"**No Mask Count:** {no_mask_count}")

# Tab for live webcam
with tabs[1]:
    st.write("Turn on your webcam to detect face masks in real-time.")
    # Dropdown for camera selection
    camera_options = [0, 1, 2, 3]  # Camera indices
    camera_index = st.selectbox("Select Camera Index", options=camera_options, index=0, key="camera_select")

    # Button to start webcam with a unique key
    if st.button("Start Webcam", key="start_webcam"):
        live_webcam(camera_index)

