import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA


CASCADE_PATH = "haarcascade_frontalface_default.xml"
VIDEO_DIR = "./video/"
DEFAULT_VIDEO = "data1.npy"
RESULTS_SAVE_DIR = "./results/"


MIN_FACE_SIZE = 100

WIDTH_FRACTION = 0.6 # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1

FPS = 14.99         #check your video frame per second !!!!!!!



def getROI(image, faceBox): 


    widthFrac = WIDTH_FRACTION
    heigtFrac = HEIGHT_FRACTION

    # Adjust bounding box
    (x, y, w, h) = faceBox
    widthOffset = int((1 - widthFrac) * w / 2)
    heightOffset = int((1 - heigtFrac) * h / 2)
    faceBoxAdjusted = (x + widthOffset, y + heightOffset,
        int(widthFrac * w), int(heigtFrac * h))

    # Segment

    (x, y, w, h) = faceBoxAdjusted
    backgrndMask = np.full(image.shape, True, dtype=bool)
    backgrndMask[y:y+h, x:x+w, :] = False
    
    (x, y, w, h) = faceBox

    roi = np.ma.array(image, mask=backgrndMask) # Masked array
    #print(roi.shape)
    return roi

# Sum of square differences between x1, x2, y1, y2 points for each ROI
def distance(roi1, roi2):
    return sum((roi1[i] - roi2[i])**2 for i in range(len(roi1)))

def getBestROI(frame, faceCascade, previousFaceBox):


    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('GRAY', gray)
    cv2.waitKey(1)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE), flags=cv2.CASCADE_SCALE_IMAGE) #HAAR_SCALE_IMAGE

    roi = None
    faceBox = None

    # If no face detected, use ROI from previous frame
    if len(faces) == 0:
        print("NoNo")
        faceBox = previousFaceBox

    # if many faces detected, use one closest to that from previous frame
    elif len(faces) > 1:
        if previousFaceBox is not None:
            # Find closest
            print("is not none")
            minDist = float("inf")
            for face in faces:
                if distance(previousFaceBox, face) < minDist:
                    faceBox = face
        else:
            # Chooses largest box by area (most likely to be true face)
            maxArea = 0
            print("else")
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    faceBox = face

    # If only one face dectected, use it!
    else:
        faceBox = faces[0]


    if faceBox is not None:

        # Show rectangle
        #(x, y, w, h) = faceBox
        #cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)

        roi = getROI(frame, faceBox)

    return faceBox, roi


# Set up video and face tracking
try:
    videoFile = sys.argv[1]
except:
    video_data = np.load(DEFAULT_VIDEO) # TODO IF no vid file is entered load default numpy array and modify the rest accordin to that



faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
print(faceCascade.empty())

colorSig = [] # Will store the average RGB color values in each frame's ROI
heartRates = [] # Will store the heart rate calculated every 1 second
previousFaceBox = None
video_path = VIDEO_DIR + DEFAULT_VIDEO
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = cap.get(cv2.CAP_PROP_FPS)
frames = []
print(f"Video Info - Total Frames: {frame_count}, Resolution: {frame_width}x{frame_height}, FPS: {fps}")
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE) #if your video isn't at portrait mode
    video_array = np.array(frame)
    #print(f"Video converted to NumPy array with shape: {video_array.shape}")
    previousFaceBox, roi = getBestROI(frame, faceCascade, previousFaceBox)

    if (roi is not None) and (np.size(roi) > 0):
        colorChannels = roi.reshape(-1, roi.shape[-1])
        avgColor = colorChannels.mean(axis=0)
        colorSig.append(avgColor)


    if np.ma.is_masked(roi):
        roi = np.where(roi.mask == True, 0, roi)
    cv2.imshow('ROI', roi)
    cv2.waitKey(1)
    frame_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)


video_data = np.array(frames)
print(np.info(video_data))
#np.save('data_vid.npy', video_data)

# to do the rest according to the homework

cap.release()
cv2.destroyAllWindows()