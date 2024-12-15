import os
import cv2
import pytesseract
from collections import Counter

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_number_plate(video_path, output_file=None):
    print(f"Absolute path: {os.path.abspath(video_path)}")
    print(f"File exists: {os.path.exists(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0
    detected_texts = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames to process.")
            break

        frame_count += 1
        print(f"Processing Frame {frame_count}...")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(thresh, config=config).strip()

        # Filter for possible number plates (10-11 alphanumeric characters)
        if 9 <= len(text) <= 11 and text.isalnum():
            print(f"Valid text detected in Frame {frame_count}: {text}")
            detected_texts.append(text)

    cap.release()

    if detected_texts:
        # Count the frequency of each detected text
        text_counts = Counter(detected_texts)
        most_common_text, _ = text_counts.most_common(1)[0]

        print(f"Most common number plate detected: {most_common_text}")

        if output_file:
            with open(output_file, 'w') as f:
                f.write(most_common_text)
            print(f"Result saved to {output_file}")
    else:
        print("No valid number plates detected.")

# Example usage
video_path = r'C:\Users\Nikhil\Desktop\VIDEO_OCR\venv\Video_OCR\vid3.mp4'
extract_number_plate(video_path, output_file='detected_number_plate.txt')