import cv2
import pytesseract
import re
import time

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Valid state codes
VALID_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ", "HR", "HP", "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OR", "PY", "PN", "RJ", "SK", "TN", "TR", "UP", "WB"
}

def is_valid_number_plate(text):
    """Validate number plate format and state code."""
    if not re.match(r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4,5}$', text):
        return False
    state_code = text[:2]
    return state_code in VALID_STATE_CODES

def correct_text(text):
    """Correct the OCR text to match the number plate format."""
    corrected_text = []
    for i, char in enumerate(text):
        if i < 2 or (4 <= i < 6):  # State code and letters
            if not char.isalpha():
                char = 'A'
        elif (2 <= i < 4) or (6 <= i):  # Digits
            if not char.isdigit():
                char = '0'
        corrected_text.append(char)
    return ''.join(corrected_text)

def preprocess_frame(frame):
    """Preprocess the frame to improve OCR accuracy."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    resized = cv2.resize(gray, (640, 480))  # Resize frame to reduce processing time
    _, thresh = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY)  # Binarize
    return thresh

def detect_number_plate_live(output_file=None):
    """Detect number plates from a live feed."""
    cap = cv2.VideoCapture('http://192.168.236.227:8080/video')  # Use mobile camera as feed

    if not cap.isOpened():
        print("Error: Unable to open the camera feed.")
        return

    # Skip frames for faster processing
    frame_count = 0
    detected_texts = set()  # Use a set to avoid duplicate detections

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to fetch frame.")
            break

        # Skip every 4 out of 5 frames
        frame_count += 1
        if frame_count % 5 != 0:
            continue

        # Preprocess frame
        start_time = time.time()
        processed_frame = preprocess_frame(frame)

        # Perform OCR
        config = '--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        text = pytesseract.image_to_string(processed_frame, config=config).strip()

        if 9 <= len(text) <= 11 and text.isalnum():
            corrected = correct_text(text)
            if is_valid_number_plate(corrected):
                if corrected not in detected_texts:
                    print(f"Valid number plate detected: {corrected}")
                    detected_texts.add(corrected)

                    if output_file:
                        with open(output_file, 'a') as f:
                            f.write(f"{corrected}\n")
                        print(f"Detected number plate saved to {output_file}")

        # Display the frame
        cv2.putText(frame, "Press 'q' to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Live Number Plate Detection", frame)

        # Show FPS
        fps = 1 / (time.time() - start_time)
        print(f"FPS: {fps:.2f}")

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the detection
detect_number_plate_live(output_file='detected_number_plate.txt')