#Arda Barış Özten - 820220303
#Berra Mutlu - 820220331
# Heart Rate Estimation from Facial Video using ICA and Signal Processing
import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.signal import detrend, butter, filtfilt, windows # Added windows for Hamming
import os

# --- Constants ---
CASCADE_PATH = "haarcascade_frontalface_default.xml"
DEFAULT_VIDEO = "data1.npy" # Default file to load if no video is provided
ROI_SIZE = (64, 64) # Define a fixed size for all ROIs for consistency, e.g., 64x64 or 128x128.
                   # Smaller might be faster but lose detail, larger is slower but retains more.

# Face detection and ROI parameters
MIN_FACE_SIZE = 100
WIDTH_FRACTION = 0.6  # Fraction of bounding box width to include in ROI
HEIGHT_FRACTION = 1.0 # Fraction of bounding box height to include in ROI

# --- Heart Rate Analysis Parameters ---
MIN_BPM = 45.0
MAX_BPM = 240.0
DEFAULT_FPS = 30.0 # Assumed FPS for .npy files


# --- Part 1: Video Processing and ROI Extraction ---

def getROI(image, faceBox):
    """
    Extracts the Region of Interest (ROI) from a face box.
    The ROI is defined by fractions of the face box's width and height.
    """
    (x, y, w, h) = faceBox
    # Calculate offset to center the ROI within the face box
    widthOffset = int((1 - WIDTH_FRACTION) * w / 2)
    heightOffset = int((1 - HEIGHT_FRACTION) * h / 2)
    
    roi_x = x + widthOffset
    roi_y = y + heightOffset
    roi_w = int(WIDTH_FRACTION * w)
    roi_h = int(HEIGHT_FRACTION * h)
    
    # Ensure ROI coordinates are within image bounds
    roi_x = max(0, roi_x)
    roi_y = max(0, roi_y)
    roi_w = min(image.shape[1] - roi_x, roi_w)
    roi_h = min(image.shape[0] - roi_y, roi_h)

    # Return the cropped ROI
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
        faceBox = previousFaceBox # Reuse previous box if no face found
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
                area = face[2] * face[3]
                if area > maxArea:
                    maxArea = area
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

        previousFaceBox, roi, display_frame = getBestROI(frame, faceCascade, previousFaceBox)

        if roi is not None and roi.size > 0:
            # Resize the ROI to our standard size to ensure all are the same.
            try:
                roi = cv2.resize(roi, ROI_SIZE)
            except cv2.error as e:
                print(f"Error resizing ROI: {e}. Skipping frame.")
                continue

            # Convert to RGB for matplotlib consistency (OpenCV reads BGR)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_frames.append(roi_rgb)
            
            # Show the tracking in real-time
            cv2.imshow('Face Tracking', display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Warning: No ROI found or ROI is empty for a frame.")

    cap.release()
    cv2.destroyAllWindows()
    
    if not roi_frames:
        raise ValueError("Could not extract any ROIs from the video.")
        
    return np.array(roi_frames), fps


# --- Part 2: Signal Analysis for Heart Rate ---

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Applies a Butterworth bandpass filter to the data.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    
    # Ensure cutoff frequencies are valid for the filter design
    if low >= high or low <= 0 or high >= 1:
        print(f"Warning: Invalid frequency range for bandpass filter: Low={low*nyq:.2f}Hz, High={high*nyq*nyq:.2f}Hz (Normalized: {low:.2f}-{high:.2f}). Skipping filter for this component.")
        return data # Return original data if filter range is invalid or too narrow

    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y


def analyze_pulse(video_data, fps):
    """
    Performs the full analysis pipeline on the ROI data to find the heart rate.
    """
    print("\n--- Starting Heart Rate Analysis ---")
    
    # --- Step 1: Spatial Pooling ---
    # Average the RGB values across all pixels in the ROI for each frame.
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
    plt.show() # Display plot immediately

    # --- Step 2: Normalization & Detrending ---
    print("Step 2: Normalizing and Detrending signals...")
    detrended_normalized_rgb = np.zeros_like(spatially_pooled_rgb, dtype=float)
    for i in range(spatially_pooled_rgb.shape[1]): # Iterate over R, G, B channels
        # Detrend first to remove linear trend (slow lighting changes)
        channel_detrended = detrend(spatially_pooled_rgb[:, i])
        # Then normalize to zero mean and unit variance
        detrended_normalized_rgb[:, i] = (channel_detrended - np.mean(channel_detrended)) / np.std(channel_detrended)
    
    # Plot the detrended and normalized RGB signals
    plt.figure(figsize=(12, 4))
    plt.plot(detrended_normalized_rgb[:, 0], 'r', label='Red Channel')
    plt.plot(detrended_normalized_rgb[:, 1], 'g', label='Green Channel')
    plt.plot(detrended_normalized_rgb[:, 2], 'b', label='Blue Channel')
    plt.title("Detrended and Normalized RGB Signal")
    plt.xlabel("Frame")
    plt.ylabel("Standardized Value")
    plt.legend()
    plt.grid(True)
    plt.show() # Display plot immediately

    # --- Step 3: Independent Component Analysis (ICA) ---
    print("Step 3: Applying Independent Component Analysis (ICA)...")
    # Using 'unit-variance' whitening ensures components are scaled similarly.
    ica = FastICA(n_components=3, random_state=42, whiten='unit-variance', max_iter=500) 
    # Use detrended_normalized_rgb as input
    source_signals_raw = ica.fit_transform(detrended_normalized_rgb) 
    
    # --- Step 3a: Bandpass Filtering of ICA Sources ---
    print("Step 3a: Applying bandpass filter to ICA signals...")
    min_hz_filter = MIN_BPM / 60.0 # Convert BPM to Hz
    max_hz_filter = MAX_BPM / 60.0
    
    source_signals_filtered = np.zeros_like(source_signals_raw)
    for i in range(source_signals_raw.shape[1]):
        source_signals_filtered[:, i] = bandpass_filter(source_signals_raw[:, i], min_hz_filter, max_hz_filter, fps)
    
    # Plot the extracted source signals (raw and filtered)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(source_signals_raw[:, 0], label='Raw Source 1')
    plt.plot(source_signals_raw[:, 1], label='Raw Source 2')
    plt.plot(source_signals_raw[:, 2], label='Raw Source 3')
    plt.title("ICA Source Signals (Raw - Before Bandpass)")
    plt.xlabel("Frame")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(source_signals_filtered[:, 0], label='Filtered Source 1')
    plt.plot(source_signals_filtered[:, 1], label='Filtered Source 2')
    plt.plot(source_signals_filtered[:, 2], label='Filtered Source 3')
    plt.title("ICA Source Signals (Bandpass Filtered)")
    plt.xlabel("Frame")
    plt.ylabel("Signal Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show() # Display plot immediately
    
    # --- Step 4: Power Spectrum Analysis ---
    # Calculate the power spectrum of each FILTERED source signal.
    print("Step 4: Calculating power spectrum of filtered source signals...")
    n_samples = len(source_signals_filtered)
    # Calculate frequencies for the FFT
    # Note: np.fft.fftfreq already gives correct positive/negative ordering
    freqs = np.fft.fftfreq(n_samples, d=1.0/fps)
    
    # Calculate power for each source (magnitude squared of FFT)
    power_spectra = np.abs(np.fft.fft(source_signals_filtered, axis=0))**2
    
    # Plot the power spectra
    plt.figure(figsize=(12, 5))
    for i in range(3):
        # Only plot positive frequencies (and corresponding power from fftshift)
        # However, for simplicity, plotting full freqs then xlim is also fine.
        plt.plot(freqs, power_spectra[:, i], label=f'Source {i+1} Power')
    plt.title("Power Spectra of ICA Sources")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    # Limit the x-axis to the specified heart rate range for better visualization
    plt.xlim([min_hz_filter, max_hz_filter])
    plt.legend()
    plt.grid(True)
    plt.show() # Display plot immediately


    # --- Step 5: Heart Rate Extraction ---
    print("Step 5: Extracting heart rate from dominant frequency...")
    
    # Find indices corresponding to the valid frequency range (positive frequencies only)
    valid_indices = np.where((freqs >= min_hz_filter) & (freqs <= max_hz_filter))
    
    all_heart_rates = []
    all_peak_powers = []
    
    # Loop through each ICA source signal
    for i in range(3):
        # Extract the power spectrum and corresponding frequencies within the valid range
        cropped_power = power_spectra[valid_indices, i].flatten()
        cropped_freqs = freqs[valid_indices].flatten()
        
        if len(cropped_power) == 0:
            print(f"Warning: No frequency components found in the valid range for source {i+1}. Skipping.")
            all_heart_rates.append(0) # Append a placeholder
            all_peak_powers.append(0) # Append a placeholder
            continue

        # Find the index of the peak power within the cropped spectrum
        peak_index = np.argmax(cropped_power)
        peak_power = cropped_power[peak_index]
        
        # Get the dominant frequency in Hz and convert to BPM
        dominant_freq_hz = cropped_freqs[peak_index]
        heart_rate_bpm = dominant_freq_hz * 60.0
        
        all_heart_rates.append(heart_rate_bpm)
        all_peak_powers.append(peak_power)
    
    # Determine the best heart rate estimate from the component with the highest BPM
    if not all_heart_rates or max(all_heart_rates) == 0:
        print("\nCould not confidently determine heart rate. No valid BPM found in the expected range.")
        final_heart_rate = None
    else:
        best_component_idx = np.argmax(all_heart_rates)
        final_heart_rate = all_heart_rates[best_component_idx]
        
        print("\n--- Results ---")
        for i in range(3):
            if all_peak_powers[i] > 0:
                print(f"Heart rate from Component {i+1}: {all_heart_rates[i]:.2f} BPM (Peak Power: {all_peak_powers[i]:.2e})")
            else:
                print(f"Heart rate from Component {i+1}: N/A (No peaks found)")
                
        print(f"\n==> Best guess is from Component {best_component_idx + 1}.")
        print(f"==> Estimated Heart Rate: {final_heart_rate:.2f} BPM")
    

    
    # Show all plots (already called after each plot)
    # plt.tight_layout() # This should be called before plt.show() if multiple subplots in one figure
    # plt.show()


# --- Main Execution ---
if __name__ == "__main__":
    video_data = None
    fps = DEFAULT_FPS

    # Add `random_state` to FastICA for reproducibility if desired
    # ica = FastICA(n_components=3, random_state=42) # Added random_state

    try:
        input_path = sys.argv[1]
        if input_path.endswith('.npy'):
            print(f"Loading data from numpy file: {input_path}")
            if not os.path.exists(input_path):
                 raise FileNotFoundError(f"Error: .npy file '{input_path}' not found.")
            video_data = np.load(input_path)
            # Make sure the loaded data is of the expected shape (frames, height, width, channels)
            if video_data.ndim != 4:
                raise ValueError(f"Loaded .npy file has unexpected dimensions: {video_data.ndim}. Expected 4D (frames, height, width, channels).")
            # If the frames in .npy are not of uniform size, resize them here too before analysis
            # This requires iterating through frames, resizing, and then stacking.
            # For simplicity, assume data1.npy already contains uniformly sized ROIs,
            # or the video processing part handles it.
            
            # If video_data comes from a file where ROIs weren't uniformly resized,
            # we need to explicitly handle it for consistency.
            resized_video_data = []
            for frame_idx in range(video_data.shape[0]):
                frame = video_data[frame_idx, ...]
                if frame.shape[:2] != ROI_SIZE: # Check if current frame's spatial size matches ROI_SIZE
                    frame = cv2.resize(frame, ROI_SIZE)
                resized_video_data.append(frame)
            video_data = np.array(resized_video_data)
            
            fps = DEFAULT_FPS # As stated, .npy doesn't carry FPS info

            print(f"Using default FPS for .npy file: {fps} Hz")
        else:
            print(f"Processing video file: {input_path}")
            video_data, fps = process_video_to_roi_array(input_path)

    except (IndexError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        print(f"No valid video file provided or file not found. Attempting to load default numpy file: {DEFAULT_VIDEO}")
        try:
            if not os.path.exists(DEFAULT_VIDEO):
                raise FileNotFoundError(f"Error: Default file {DEFAULT_VIDEO} not found. Please provide a video file path or place 'data1.npy' in the script's directory.")
            video_data = np.load(DEFAULT_VIDEO)
            # Apply ROI_SIZE resizing for .npy data as well, for consistency
            resized_video_data = []
            for frame_idx in range(video_data.shape[0]):
                frame = video_data[frame_idx, ...]
                if frame.shape[:2] != ROI_SIZE:
                    frame = cv2.resize(frame, ROI_SIZE) # Ensure it's resized to the standard ROI_SIZE
                resized_video_data.append(frame)
            video_data = np.array(resized_video_data)

            fps = DEFAULT_FPS
            print(f"Successfully loaded {DEFAULT_VIDEO}. Using default FPS: {fps} Hz")
        except FileNotFoundError as e:
            print(f"Fatal Error: {e}. Exiting.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when loading default .npy file: {e}. Exiting.")
            sys.exit(1)

    if video_data is not None and video_data.size > 0:
        print("\n--- Video Data Info (After Preprocessing) ---")
        print(f"Shape: {video_data.shape}")
        print(f"Data Type: {video_data.dtype}")
        print(f"Min/Max pixel values: {video_data.min()}/{video_data.max()}")
        analyze_pulse(video_data, fps)
    else:
        print("Could not load or process video data. Exiting.")