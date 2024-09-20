import os
import cv2
import numpy as np
from tqdm import tqdm

def detect_face(frame):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        return frame[y:y+h, x:x+w]
    return None

def process_video(input_path, output_dir, target_size=(224, 224)):
    cap = cv2.VideoCapture(input_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in tqdm(range(frame_count), desc=f"Processing {os.path.basename(input_path)}"):
        ret, frame = cap.read()
        if not ret:
            break
        
        face_frame = detect_face(frame)
        if face_frame is not None:
            face_frame = cv2.resize(face_frame, target_size)
            output_path = os.path.join(output_dir, f"frame_{i:04d}.jpg")
            cv2.imwrite(output_path, face_frame)
    
    cap.release()

def process_dataset(input_dir, output_dir):
    for class_name in ['Stress', 'Normal']:
        input_class_dir = os.path.join(input_dir, class_name)
        output_class_dir = os.path.join(output_dir, class_name)
        
        for video_file in os.listdir(input_class_dir):
            if video_file.endswith('.mp4'):
                input_path = os.path.join(input_class_dir, video_file)
                video_name = os.path.splitext(video_file)[0]
                output_video_dir = os.path.join(output_class_dir, video_name)
                process_video(input_path, output_video_dir)

if __name__ == "__main__":
    input_dir = "Clips"
    output_dir = "rlddi"
    process_dataset(input_dir, output_dir)