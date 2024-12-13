import os
import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text_from_video(video_path, output_file=None):
    # inki wajah se speed slow hai, optimize karenge baad mein
    print(f"Absolute path: {os.path.abspath(video_path)}")
    print(f"File exists: {os.path.exists(video_path)}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0
    results = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No more frames to process.")
            break

        frame_count += 1
        print(f"Processing Frame {frame_count}...")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        # Save karna hai to, every frame
        # cv2.imwrite(f'processed_frame_{frame_count}.jpg', thresh)

        config = '--psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
        text = pytesseract.image_to_string(thresh, config=config).strip()
        
        if text:
            print(f"Detected text in Frame {frame_count}: {text}")
            results.append((frame_count, text))

    cap.release()

    if output_file:
        with open(output_file, 'w') as f:
            for frame_number, detected_text in results:
                f.write(f"Frame {frame_number}: {detected_text}\n")
        print(f"Results saved to {output_file}")

video_path = r'C:\Users\Nikhil\Desktop\VIDEO_OCR\venv\Video_OCR\vid2.mp4'
extract_text_from_video(video_path, output_file='detected_texts.txt')