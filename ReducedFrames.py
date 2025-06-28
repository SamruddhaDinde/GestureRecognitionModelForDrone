import cv2
import os

video_path = 'FinalRoomData.mp4'
output_folder = 'FinalFramesRoom'
frame_interval = 2  

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_num = 0
saved_num = 3860

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_num % frame_interval == 0:
        frame_filename = os.path.join(output_folder, f"frame_{saved_num:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved_num += 1
    frame_num += 1

cap.release()
print(f"Extracted {saved_num} frames to {output_folder}")
