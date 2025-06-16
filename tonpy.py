import cv2
import numpy as np
import sys

# Usage: python video_to_npy.py input_video.mp4 output.npy

if len(sys.argv) < 3:
    print("Usage: python video_to_npy.py input_video.mp4 output.npy")
    sys.exit(1)

video_path = sys.argv[1]
output_npy = sys.argv[2]
ROI_SIZE = (64, 64)  # Match your Heart2Faceb.py ROI_SIZE

cap = cv2.VideoCapture(video_path)
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Resize to ROI_SIZE and convert to RGB
    frame_resized = cv2.resize(frame, ROI_SIZE)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)

cap.release()

video_array = np.array(frames)
np.save(output_npy, video_array)
print(f"Saved {len(frames)} frames to {output_npy} with shape {video_array.shape}")