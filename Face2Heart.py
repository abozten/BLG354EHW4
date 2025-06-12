import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import os

# --- Constants ---
CASCADE_PATH = "haarcascade_frontalface_default.xml"
DEFAULT_VIDEO = "berra.mp4" # Default file to load if no video is provided

# Face detection and ROI parameters
MIN_FACE_SIZE = 100
WIDTH_FRACTION = 0.6  # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1.0 # Fraction of bounding box height to include in ROI

# --- Heart Rate Analysis Parameters ---
# As per the homework, the plausible heart rate range is 45 to 240 BPM.
MIN_BPM = 45.0
MAX_BPM = 240.0
# We assume a default FPS for .npy files since this info isn't stored in the array.
# The homework uses 14.99 in the original script and mentions 30Hz as typical.
# We will use 30 as a more standard default.
DEFAULT_FPS = 30.0


# --- Part 1: Video Processing and ROI Extraction ---

def getROI(image, faceBox):
    """
    Extracts the Region of Interest (ROI) from a face box.
    The ROI is defined by fractions of the face box's width and height.
    """
    # Adjust bounding box to select a smaller ROI (e.g., forehead and cheeks)
    (x, y, w, h) = faceBox
    widthOffset = int((1 - WIDTH_FRACTION) * w / 2)
    heightOffset = int((1 - HEIGHT_FRACTION) * h / 2)
    roi_x = x + widthOffset
    roi_y = y + heightOffset
    roi_w = int(WIDTH_FRACTION * w)
    roi_h = int(HEIGHT_FRACTION * h)
    
    # Create a mask to blank out everything outside the ROI
    mask = np.full(image.shape[:2], False, dtype=bool)
    mask[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w] = True
    
    # Apply mask
    roi = np.zeros_like(image)
    roi[mask] = image[mask]
    
    # Return the cropped ROI for display, and the masked full image for processing
    # Note: we return the cropped version for building the video_data array
    return image[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]


def distance(box1, box2):
    """Calculates the sum of square differences between two bounding boxes."""
    return sum((box1[i] - box2[i])**2 for i in range(len(box1)))


def getBestROI(frame, faceCascade, previousFaceBox):
    """
    Detects faces in a frame and returns the best ROI.
    - If one face is found, use it.
    - If multiple faces are found, use the one closest to the previous frame's face.
    - If no faces are found, re-use the previous frame's face box.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    faceBox = None
    if len(faces) == 0:
        faceBox = previousFaceBox
    elif len(faces) > 1:
        if previousFaceBox is not None:
            # Find face closest to the one in the previous frame
            minDist = float("inf")
            for face in faces:
                d = distance(previousFaceBox, face)
                if d < minDist:
                    minDist = d
                    faceBox = face
        else:
            # If no previous face, choose the largest one
            maxArea = 0
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    maxArea = face[2] * face[3]
                    faceBox = face
    else:
        # Only one face was detected
        faceBox = faces[0]

    roi = None
    if faceBox is not None:
        roi = getROI(frame, faceBox)
        # For visualization, draw rectangle on the original frame
        (x, y, w, h) = faceBox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)


    return faceBox, roi, frame


def process_video_to_roi_array(video_path):
    """
    Reads a video file, performs face tracking, extracts ROIs,
    and returns a 4D numpy array of the ROIs and the video's FPS.
    """
    if not os.path.exists(CASCADE_PATH):
        raise FileNotFoundError(f"Haar Cascade file not found at: {CASCADE_PATH}")
    faceCascade = cv2.CascadeClassifier(CASCADE_PATH)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print(f"Warning: Could not get FPS from video. Using default: {DEFAULT_FPS} Hz")
        fps = DEFAULT_FPS

    print(f"Video Info - FPS: {fps:.2f}")

    roi_frames = []
    previousFaceBox = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Some videos might be rotated
        # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        previousFaceBox, roi, display_frame = getBestROI(frame, faceCascade, previousFaceBox)

        if roi is not None and roi.size > 0:
            # Convert to RGB for matplotlib consistency
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_frames.append(roi_rgb)
            
            # Show the tracking in real-time
            cv2.imshow('Face Tracking', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Warning: No ROI found for a frame.")

    cap.release()
    cv2.destroyAllWindows()
    
    if not roi_frames:
        raise ValueError("Could not extract any ROIs from the video.")
        
    return np.array(roi_frames), fps


# --- Part 2: Signal Analysis for Heart Rate ---

def analyze_pulse(video_data, fps):
    """
    Performs the full analysis pipeline on the ROI data to find the heart rate.
    """
    print("\n--- Starting Heart Rate Analysis ---")
    
    # --- Step 1: Spatial Pooling ---
    # Average the RGB values across all pixels in the ROI for each frame.
    # This reduces the 4D (frames, height, width, channels) array to 2D (frames, channels).
    print("Step 1: Performing spatial pooling on ROI data...")
    spatially_pooled_rgb = np.mean(video_data, axis=(1, 2))
    
    # Plot the raw, spatially pooled RGB signal
    plt.figure(figsize=(12, 4))
    plt.plot(spatially_pooled_rgb[:, 0], 'r', label='Red Channel')
    plt.plot(spatially_pooled_rgb[:, 1], 'g', label='Green Channel')
    plt.plot(spatially_pooled_rgb[:, 2], 'b', label='Blue Channel')
    plt.title("Spatially Pooled RGB Signal (Raw)")
    plt.xlabel("Frame")
    plt.ylabel("Average Pixel Value")
    plt.legend()
    plt.grid(True)
    
    # --- Step 2: Normalization ---
    # Standardize each channel to have a mean of 0 and a standard deviation of 1.
    print("Step 2: Normalizing signals...")
    mean_rgb = np.mean(spatially_pooled_rgb, axis=0)
    std_rgb = np.std(spatially_pooled_rgb, axis=0)
    normalized_rgb = (spatially_pooled_rgb - mean_rgb) / std_rgb
    
    # Plot the normalized RGB signals
    plt.figure(figsize=(12, 4))
    plt.plot(normalized_rgb[:, 0], 'r', label='Red Channel')
    plt.plot(normalized_rgb[:, 1], 'g', label='Green Channel')
    plt.plot(normalized_rgb[:, 2], 'b', label='Blue Channel')
    plt.title("Normalized RGB Signal")
    plt.xlabel("Frame")
    plt.ylabel("Standardized Value")
    plt.legend()
    plt.grid(True)

    # --- Step 3: Independent Component Analysis (ICA) ---
    # Use FastICA to unmix the signals into independent sources.
    print("Step 3: Applying Independent Component Analysis (ICA)...")
    ica = FastICA(n_components=3, random_state=0, whiten='unit-variance')
    source_signals = ica.fit_transform(normalized_rgb)
    
    # Plot the extracted source signals
    plt.figure(figsize=(12, 4))
    plt.plot(source_signals[:, 0], label='Source 1')
    plt.plot(source_signals[:, 1], label='Source 2 (often the pulse)')
    plt.plot(source_signals[:, 2], label='Source 3')
    plt.title("ICA Source Signals")
    plt.xlabel("Frame")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid(True)
    
    # --- Step 4: Power Spectrum Analysis ---
    # Calculate the power spectrum of each source signal to find dominant frequencies.
    print("Step 4: Calculating power spectrum of source signals...")
    n_samples = len(source_signals)
    # Calculate frequencies for the FFT
    freqs = np.fft.fftfreq(n_samples, d=1.0/fps)
    
    # Calculate power for each source
    power_spectra = np.abs(np.fft.fft(source_signals, axis=0))**2
    
    # Plot the power spectra
    plt.figure(figsize=(12, 5))
    for i in range(3):
        plt.plot(freqs, power_spectra[:, i], label=f'Source {i+1} Power')
    plt.title("Power Spectra of ICA Sources")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    # Limit the x-axis to the specified heart rate range for better visualization
    plt.xlim([MIN_BPM / 60, MAX_BPM / 60])
    plt.legend()
    plt.grid(True)

    # --- Step 5: Heart Rate Extraction ---
    # Find the dominant frequency in the plausible range [0.75 Hz, 4 Hz].
    print("Step 5: Extracting heart rate from dominant frequency...")
    min_hz = MIN_BPM / 60.0
    max_hz = MAX_BPM / 60.0
    
    # Find indices corresponding to the valid frequency range
    valid_indices = np.where((freqs >= min_hz) & (freqs <= max_hz))
    
    all_heart_rates = []
    all_peak_powers = []
    
    for i in range(3):
        cropped_power = power_spectra[valid_indices, i].flatten()
        cropped_freqs = freqs[valid_indices].flatten()
        
        if len(cropped_power) == 0:
            print(f"Warning: No frequency components found in the valid range for source {i+1}.")
            all_heart_rates.append(0)
            all_peak_powers.append(0)
            continue

        # Find the peak power in the cropped spectrum
        peak_index = np.argmax(cropped_power)
        peak_power = cropped_power[peak_index]
        
        # Find the corresponding frequency and convert to BPM
        dominant_freq_hz = cropped_freqs[peak_index]
        heart_rate_bpm = dominant_freq_hz * 60.0
        
        all_heart_rates.append(heart_rate_bpm)
        all_peak_powers.append(peak_power)
    
    # The best estimate is from the component with the highest peak power
    best_component_idx = np.argmax(all_peak_powers)
    final_heart_rate = all_heart_rates[best_component_idx]
    
    print("\n--- Results ---")
    for i in range(3):
        print(f"Heart rate from Component {i+1}: {all_heart_rates[i]:.2f} BPM (Peak Power: {all_peak_powers[i]:.2e})")
        
    print(f"\n==> Best guess is from Component {best_component_idx + 1}.")
    print(f"==> Estimated Heart Rate: {final_heart_rate:.2f} BPM")
    
    # --- Step 6: Further Improvements ---
    print("\n--- Suggestions for Further Improvements ---")
    print("1. Detrending: Apply a detrending algorithm (e.g., `scipy.signal.detrend`) to the\n"
          "   normalized signals before ICA to remove slow-moving lighting changes.")
    print("2. Bandpass Filtering: Filter the ICA source signals to only keep frequencies\n"
          "   within the valid heart rate range (e.g., 0.75-4Hz). This can remove noise\n"
          "   and improve the accuracy of the peak finding.")
    print("3. Sliding Window Analysis: Instead of one value for the whole video, calculate\n"
          "   heart rate over a moving window (e.g., 30 seconds) to track changes over time.")
    print("4. Advanced ROI Selection: Instead of a simple box, automatically segment physiologically\n"
          "   relevant areas like the forehead, which often yield a cleaner signal.")

    # Show all plots
    plt.tight_layout()
    plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    video_data = None
    fps = DEFAULT_FPS

    try:
        input_path = sys.argv[1]
        if input_path.endswith('.npy'):
            print(f"Loading data from numpy file: {input_path}")
            if not os.path.exists(input_path):
                 raise FileNotFoundError
            video_data = np.load(input_path)
            fps = DEFAULT_FPS
            print(f"Using default FPS for .npy file: {fps} Hz")
        else:
            print(f"Processing video file: {input_path}")
            video_data, fps = process_video_to_roi_array(input_path)

    except (IndexError, FileNotFoundError):
        print(f"No valid video file provided or file not found.")
        print(f"Attempting to load default numpy file: {DEFAULT_VIDEO}")
        try:
            video_data = np.load(DEFAULT_VIDEO)
            fps = DEFAULT_FPS
            print(f"Successfully loaded {DEFAULT_VIDEO}. Using default FPS: {fps} Hz")
        except FileNotFoundError:
            print(f"Error: Default file {DEFAULT_VIDEO} not found. Please provide a video file path or place 'data1.npy' in the script's directory.")
            sys.exit(1)

    if video_data is not None and video_data.size > 0:
        # np.info(video_data) # As requested in HW
        print("\n--- Video Data Info ---")
        print(f"Shape: {video_data.shape}")
        print(f"Data Type: {video_data.dtype}")
        print(f"Min/Max values: {video_data.min()}/{video_data.max()}")
        analyze_pulse(video_data, fps)
    else:
        print("Could not load or process video data. Exiting.")