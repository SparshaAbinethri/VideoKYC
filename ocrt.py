import cv2
import pytesseract
from PIL import Image

# If using Windows, specify the path to the Tesseract executable
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def preprocess_frame(frame):
    """
    Preprocess the frame for better text detection.
    """
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Optional: Apply thresholding for better text extraction
    _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return binary

def extract_text(frame):
    """
    Extract text from a preprocessed frame using Tesseract.
    """
    # Convert the frame to a PIL Image
    pil_image = Image.fromarray(frame)
    # Perform OCR using Tesseract
    text = pytesseract.image_to_string(pil_image)
    return text

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the webcam.")
    exit()

print("Press 'q' to quit the live stream.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from webcam.")
        break

    # Preprocess the frame for Tesseract
    processed_frame = preprocess_frame(frame)

    # Extract text from the processed frame
    detected_text = extract_text(processed_frame)

    # Display the detected text on the original frame
    cv2.putText(frame, f"Text: {detected_text.strip()[:50]}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show the live webcam feed with overlayed text
    cv2.imshow("Live Text Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
