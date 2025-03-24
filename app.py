import streamlit as st
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from PIL import Image
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceDetectionModule import FaceDetector
import random
import numpy as np
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import time
import pytesseract
pytesseract.pytesseract.tesseract_cmd ="/System/Volumes/Data/opt/homebrew/Cellar/tesseract/5.5.0/bin/tesseract"
import tensorflow as tf
from tensorflow.keras.models import load_model

def create_embedding_model():
    inputs = Input(shape=(224, 224, 3))
    x = Conv2D(16, (3, 3), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(64)(x)  # Output layer with 64-dim embedding
    model = Model(inputs, outputs)
    return model

# Preprocess image to ensure compatibility with the model
def preprocess_image(image):
    # Convert to RGB if not already
    if image.mode != "RGB":
        image = image.convert("RGB")
    # Resize image to match the model input shape
    image = image.resize((224, 224))
    # Normalize the pixel values to [0, 1]
    image_array = np.array(image) / 255.0
    # Ensure the input has the correct shape
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Get embedding from the model
def get_embedding(image):
    try:
        processed_image = preprocess_image(image)
        embedding = model.predict(processed_image)
        return embedding
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

# Compare two embeddings
def compare_faces(embedding1, embedding2):
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity[0][0]

# Initialize the embedding model
model = create_embedding_model()
model.compile(optimizer='adam', loss='mse')  # Compile the model

def process_frame_for_ocr(frame):
    """Convert a video frame to a format suitable for OCR."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    _, thresholded = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # Binarize
    return thresholded

def extract_text_from_frame(frame):
    """Use pytesseract to extract text from a video frame."""
    processed_frame = process_frame_for_ocr(frame)
    text = pytesseract.image_to_string(processed_frame)
    return text.strip()


model_path = "emotion_detection_model.h5"
emotion_model = load_model(model_path)

# Define emotion labels (example labels, replace with actual model labels)
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# Function to preprocess frames
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (48, 48))
    if len(resized_frame.shape) == 2:  # If grayscale
        resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_GRAY2RGB)
    elif resized_frame.shape[2] == 1:  # If single channel
        resized_frame = np.repeat(resized_frame, 3, axis=2)

    normalized_frame = resized_frame / 255.0  # Normalize pixel values
    reshaped_frame = np.expand_dims(normalized_frame, axis=0)  # Add batch dimension
    return reshaped_frame


# Initialize session state for navigation and verified sections if not already set
if "section" not in st.session_state:
    st.session_state.section = 0

if "verified_sections" not in st.session_state:
    st.session_state.verified_sections = [False] * 4  # A list to track verification status of each section

# Define section names
section_names = [
    "Face Verification",
    "Hand Verification",
    "ID Detection",
    "Emotion Detection"
]

# Check if all sections are verified
if all(st.session_state.verified_sections):
    st.success("Your KYC is Approved!")
    st.stop()  # Stop further execution of the app

# Sidebar Navigation - Section buttons with conditional styling
st.sidebar.header("Settings")
for i, name in enumerate(section_names):
    if st.session_state.verified_sections[i]:
        # Render a green button-like element for verified sections
        st.sidebar.markdown(
            f"<button disabled style='background-color: green; color: white; "
            f"border: none; padding: 10px 20px; text-align: center; "
            f"display: inline-block; font-size: 16px; border-radius: 5px;'>"
            f"{name} (Verified)</button>",
            unsafe_allow_html=True
        )
    else:
        # Standard Streamlit button for unverified sections
        if st.sidebar.button(name, key=f"section_{i}_button"):
            st.session_state.section = i

# Main window logic: Define next and previous buttons
def next_section():
    st.session_state.section = (st.session_state.section + 1) % len(section_names)

def prev_section():
    st.session_state.section = (st.session_state.section - 1) % len(section_names)

# Page Title
st.title("Video KYC Verification")

# Create two columns for buttons with a spacer
col1, spacer, col2 = st.columns([1, 0.2, 1])

with col1:
    if st.button("Previous", key="prev_button"):
        prev_section()

with col2:
    if st.button("Next", key="next_button"):
        next_section()

# Display content for each section based on the current section
st.header(f"Section {st.session_state.section + 1}: {section_names[st.session_state.section]}")

if st.session_state.section == 0:
    st.header("Face Verification")
    st.write("Upload your ID image and ensure your face is detected on the live feed for verification.")

    # Step 1: Upload ID Image
    id_image_file = st.file_uploader("Upload ID Image", type=["jpg", "jpeg", "png"])
    
    if id_image_file:
        # Load and display the uploaded ID image
        id_image = Image.open(id_image_file)
        st.image(id_image, caption="Uploaded ID Image", use_column_width=True)
        
        # Step 2: Get embedding for the uploaded ID image
        id_embedding = None
        try:
            id_embedding = get_embedding(id_image)
        except Exception as e:
            st.error(f"Error in processing ID image: {e}")

    # Step 3: Initialize live video feed for face verification
    run_face_verification = st.checkbox("Start Face Verification", value=False)
    frame_window = st.image([])  # Placeholder for displaying video frames

    if id_image_file and id_embedding is not None and run_face_verification:
        cap = cv2.VideoCapture(0)
        verified = False

        while run_face_verification:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera!")
                break

            # Convert frame to PIL image for compatibility
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Step 4: Get embedding for the live frame and compare
            try:
                live_embedding = get_embedding(pil_frame)
                similarity_score = compare_faces(id_embedding, live_embedding)

                # Display similarity score on the frame
                cv2.putText(
                    frame, 
                    f"Similarity: {similarity_score:.2f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 0), 
                    2
                )

                if similarity_score > 0.9:  # Set threshold for verification
                    verified = True
                    st.success("Face Verification Successful!")
                    st.session_state.verified_sections[0] = True  # Mark Section 1 as verified
                    run_face_verification = False  # Stop detection loop

            except Exception as e:
                st.error(f"Error in processing live frame: {e}")

            # Convert the frame to RGB and display in Streamlit
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_window.image(img_rgb)

        cap.release()
        cv2.destroyAllWindows()

    elif st.session_state.verified_sections[0]:
        st.success("Face Verification already completed!")


elif st.session_state.section == 1:
    left_hand_number, right_hand_number = random.randint(0, 5), random.randint(0, 5)

    condition_met = False
    run_app = st.checkbox(
        "Start Hand Detection", 
        value=False, 
        disabled=condition_met
    )
    st.header(f"Left Hand: {left_hand_number}          Right Hand: {right_hand_number}")
    st.write("Place your hand in front of the camera for detection.")
    FRAME_WINDOW = st.image([])
    feedback_text = st.empty()

    detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5)

    cap = cv2.VideoCapture(0)  # Start video capture

    while run_app:
        success, img = cap.read()
        if not success:
            break 
        hands, img = detector.findHands(img, draw=True, flipType=True)                    
        if hands:
            right_hand_fingers = None
            left_hand_fingers = None
            for hand in hands:
                hand_type = hand["type"]
                fingers = detector.fingersUp(hand)
                finger_count = fingers.count(1)
                if hand_type == "Right":
                    right_hand_fingers = finger_count
                    cv2.putText(img, f'Right Hand: {right_hand_fingers}', (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                elif hand_type == "Left":
                    left_hand_fingers = finger_count
                    cv2.putText(img, f'Left Hand: {left_hand_fingers}', (280, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            if right_hand_fingers is not None and left_hand_fingers is not None:
                if right_hand_fingers == right_hand_number and left_hand_fingers == left_hand_number:
                    #st.info("Hand Verified. Verification Complete!")
                    st.session_state.verified_sections[1] = True
                    condition_met = True
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for Streamlit
        FRAME_WINDOW.image(img_rgb)
        if condition_met:
            st.success("Hand Verified. Verification Complete!")
            time.sleep(3)
            break
    cap.release()  # Release the video capture resource

elif st.session_state.section == 2:
    if st.button("Verify Section 2"):
        st.session_state.verified_sections[2] = True
    input_word = st.text_input("Enter the word to detect in the video feed", placeholder="Enter your word here...")

    # Add a checkbox to start verification
    start_verification = st.checkbox("Start Verification", disabled=not bool(input_word))

    if not input_word:
        st.warning("Please enter a word to enable verification.")
    elif start_verification:
        st.info(f"Looking for the word: '{input_word}' in the live video feed.")
        
        # Start video capture and live detection
        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])  # Placeholder for video feed
        detected = False
        
        while not detected:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access the camera. Please check your setup.")
                break
            
            # Extract text from the frame
            text_in_frame = extract_text_from_frame(frame)
            st.write(f"Detected Text: {text_in_frame}")  # Optional: Display detected text
            
            # Check if the input word is in the detected text
            if input_word.lower() in text_in_frame.lower():
                detected = True
                st.success(f"Word '{input_word}' found in the video feed!")
                break
            
            # Display the video feed in the app
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

            # Stop the loop if the word is not found within a reasonable time or user intervention
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Allow user to quit with 'q'
                break
        
        cap.release()
        cv2.destroyAllWindows()  

# Mark Section 3 as verified if detected
    if start_verification and detected:
        st.session_state.verified_sections[2] = True

elif st.session_state.section == 3:
    st.write("Real-time Emotion Detection")
    detected_emotions=[]

    # Add a checkbox to start emotion detection
    start_emotion_detection = st.checkbox("Start Emotion Detection")

    if start_emotion_detection:
        cap = cv2.VideoCapture(0)
        FRAME_WINDOW = st.image([])
        start_time = time.time()

        while start_emotion_detection:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            # Preprocess the frame for emotion detection
            preprocessed_frame = preprocess_frame(frame)
            predictions = emotion_model.predict(preprocessed_frame)
            emotion_index = np.argmax(predictions)
            detected_emotion = emotion_labels[emotion_index]

            detected_emotions.append(detected_emotion)

            # Display the detected emotion on the frame
            cv2.putText(
                frame,
                f"Emotion: {detected_emotion}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            if time.time() - start_time >= 10:
                break
            # Convert the frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb)

        cap.release()
        #cv2.destroyAllWindows()
        print(detected_emotions)

        # Verify if happy or sad is in the detected emotions
        if "Sad" in detected_emotions or "Disgust" in detected_emotions:
            st.error("Verification Failed")
            st.session_state.verified_sections[3] = False
        else:
            st.success(f"Verification Successful")
            st.session_state.verified_sections[3] = True
