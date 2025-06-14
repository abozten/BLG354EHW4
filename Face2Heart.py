import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from scipy.signal import detrend, butter, filtfilt
import os

# --- Constants ---
CASCADE_PATH = "haarcascade_frontalface_default.xml"
DEFAULT_VIDEO = "data1.npy"

# Face detection and ROI parameters
MIN_FACE_SIZE = 100
ROI_SIZE = (128, 128)  # A fixed size for all ROIs, e.g., 128x128 pixels

# Heart Rate Analysis Parameters
MIN_BPM = 45.0   # Corresponds to 0.75 Hz
MAX_BPM = 240.0  # Corresponds to 4.0 Hz
DEFAULT_FPS = 30.0

# --- Part 1: Video Processing and ROI Extraction ---

def get_forehead_roi(image, faceBox):
    """
    Extracts the forehead region from a face box.
    This is often a more stable region for pulse detection.
    """
    (x, y, w, h) = faceBox
    
    # Define forehead area: top 25% of the face box, middle 60% width
    forehead_x = x + int(w * 0.2)
    forehead_y = y + int(h * 0.05) # Start a little below the top edge
    forehead_w = int(w * 0.6)
    forehead_h = int(h * 0.25)
    
    return image[forehead_y : forehead_y + forehead_h, forehead_x : forehead_x + forehead_w]

def distance(box1, box2):
    """Calculates the sum of square differences between two bounding boxes."""
    return sum((box1[i] - box2[i])**2 for i in range(len(box1)))

def getBestROI(frame, faceCascade, previousFaceBox):
    """
    Detects faces in a frame and returns the best FOREHEAD ROI.
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
            minDist = float("inf")
            for face in faces:
                d = distance(previousFaceBox, face)
                if d < minDist:
                    minDist = d
                    faceBox = face
        else:
            maxArea = 0
            for face in faces:
                if (face[2] * face[3]) > maxArea:
                    maxArea = face[2] * face[3]
                    faceBox = face
    else:
        faceBox = faces[0]

    roi = None
    if faceBox is not None:
        roi = get_forehead_roi(frame, faceBox)
        # For visualization, draw both face and forehead boxes
        (x, y, w, h) = faceBox
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        forehead_x = x + int(w * 0.2)
        forehead_y = y + int(h * 0.05)
        forehead_w = int(w * 0.6)
        forehead_h = int(h * 0.25)
        cv2.rectangle(frame, (forehead_x, forehead_y), (forehead_x + forehead_w, forehead_y + forehead_h), (0, 255, 0), 2)

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
            roi = cv2.resize(roi, ROI_SIZE)
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_frames.append(roi_rgb)
            
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

# --- (Keep all other functions and imports the same) ---

def analyze_pulse(video_data, fps):
    """
    Performs the full analysis pipeline on the ROI data to find the heart rate.
    This DEFINITIVE version has ACCURATE component selection and PDF-COMPLIANT plots.
    """
    print("\n--- Starting Heart Rate Analysis (Definitive Version) ---")
    
    # === Step 1-3: Signal Preparation ===
    print("Steps 1-3: Pooling, Detrending, and Normalizing...")
    spatially_pooled_rgb = np.mean(video_data, axis=(1, 2))
    detrended_rgb = np.zeros_like(spatially_pooled_rgb)
    for i in range(spatially_pooled_rgb.shape[1]):
        detrended_rgb[:, i] = detrend(spatially_pooled_rgb[:, i])
    mean_rgb = np.mean(detrended_rgb, axis=0)
    std_rgb = np.std(detrended_rgb, axis=0)
    normalized_rgb = (detrended_rgb - mean_rgb) / (std_rgb + 1e-6)

    # === Step 4: ICA ===
    print("Step 4: Applying Independent Component Analysis (ICA)...")
    ica = FastICA(n_components=3, random_state=0, whiten='unit-variance', max_iter=1000)
    source_signals = ica.fit_transform(normalized_rgb)

    # === Step 5: Power Spectrum Calculation for ALL signals ===
    print("Step 5: Calculating power spectra for all signals...")
    n_samples = len(normalized_rgb)
    freqs = np.fft.fftfreq(n_samples, d=1.0/fps)
    green_power_spectrum = np.abs(np.fft.fft(normalized_rgb[:, 1]))**2
    ica_power_spectra = np.abs(np.fft.fft(source_signals, axis=0))**2
    
    # === Plotting Window 1: PDF Compliance Plots ===
    print("Generating diagnostic plots required by PDF...")
    plt.figure(figsize=(12, 8))
    
    # Plot 1: ICA Source Signals (Time-Domain)
    plt.subplot(2, 1, 1)
    plt.title("Extracted ICA Source Signals (Time-Domain)")
    plt.plot(source_signals[:, 0], label='Source 1')
    plt.plot(source_signals[:, 1], label='Source 2')
    plt.plot(source_signals[:, 2], label='Source 3')
    plt.xlabel("Frame")
    plt.ylabel("Signal Amplitude")
    plt.grid(True)
    plt.legend()
    
    # Plot 2: Power Spectra of all ICA Sources
    min_hz = MIN_BPM / 60.0
    max_hz = MAX_BPM / 60.0
    plt.subplot(2, 1, 2)
    plt.title("Power Spectra of all ICA Sources")
    plt.plot(freqs, ica_power_spectra[:, 0], label='Source 1 Power')
    plt.plot(freqs, ica_power_spectra[:, 1], label='Source 2 Power')
    plt.plot(freqs, ica_power_spectra[:, 2], label='Source 3 Power')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim(min_hz, max_hz)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()

    # === Step 6: Intelligent Component Selection ===
    print("Step 6: Selecting best component via Green Channel correlation...")
    best_component_idx = -1
    max_correlation = -1
    valid_indices = np.where((freqs >= min_hz) & (freqs <= max_hz))
    
    for i in range(source_signals.shape[1]):
        correlation = np.corrcoef(
            green_power_spectrum[valid_indices], 
            ica_power_spectra[valid_indices, i]
        )[0, 1]
        print(f"Component {i+1} correlation with Green channel: {correlation:.4f}")
        if correlation > max_correlation:
            max_correlation = correlation
            best_component_idx = i

    print(f"==> Selected Component {best_component_idx + 1} as most likely pulse signal.")

    # === Step 7: Filtering and Final Extraction ===
    print("Step 7: Filtering chosen component and extracting final heart rate...")
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low, high = lowcut / nyq, highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    filtered_pulse_signal = bandpass_filter(source_signals[:, best_component_idx], min_hz, max_hz, fps)
    final_power_spectrum = np.abs(np.fft.fft(filtered_pulse_signal))**2
    
    peak_index_in_cropped = np.argmax(final_power_spectrum[valid_indices])
    dominant_freq_hz = freqs[valid_indices][peak_index_in_cropped]
    final_heart_rate = dominant_freq_hz * 60.0

    print("\n--- Results ---")
    print(f"==> Final Estimated Heart Rate: {final_heart_rate:.2f} BPM")
    
    # === Plotting Window 2: Final Result Verification ===
    plt.figure(figsize=(10, 5))
    plt.title("Final Result: Power Spectrum of Chosen Component")
    plt.plot(freqs, final_power_spectrum, label=f'Chosen Component ({best_component_idx+1}) Power')
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim([min_hz, max_hz])
    plt.axvline(x=dominant_freq_hz, color='r', linestyle='--', label=f'Detected Peak ({final_heart_rate:.2f} BPM)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Show all generated plot windows
    plt.show()
    """
    Performs the full analysis pipeline on the ROI data to find the heart rate.
    This FINAL version includes intelligent component selection based on the green channel.
    """
    print("\n--- Starting Heart Rate Analysis (Intelligent Selection) ---")
    
    # Step 1: Spatial Pooling
    print("Step 1: Performing spatial pooling on ROI data...")
    spatially_pooled_rgb = np.mean(video_data, axis=(1, 2))
    
    # Step 2: Detrending
    print("Step 2: Detrending signals...")
    detrended_rgb = np.zeros_like(spatially_pooled_rgb)
    for i in range(spatially_pooled_rgb.shape[1]):
        detrended_rgb[:, i] = detrend(spatially_pooled_rgb[:, i])
        
    # Step 3: Normalization
    print("Step 3: Normalizing signals...")
    mean_rgb = np.mean(detrended_rgb, axis=0)
    std_rgb = np.std(detrended_rgb, axis=0)
    normalized_rgb = (detrended_rgb - mean_rgb) / (std_rgb + 1e-6)

    # Step 4: Independent Component Analysis (ICA)
    print("Step 4: Applying Independent Component Analysis (ICA)...")
    ica = FastICA(n_components=3, random_state=0, whiten='unit-variance', max_iter=1000)
    source_signals = ica.fit_transform(normalized_rgb)

    # --- Step 5: Intelligent Component Selection ---
    print("Step 5: Selecting best component based on Green Channel correlation...")
    
    # Calculate power spectrum for all signals
    n_samples = len(normalized_rgb)
    freqs = np.fft.fftfreq(n_samples, d=1.0/fps)
    
    # Get power spectrum of the original GREEN channel
    green_power_spectrum = np.abs(np.fft.fft(normalized_rgb[:, 1]))**2
    
    # Get power spectra of the ICA source signals
    ica_power_spectra = np.abs(np.fft.fft(source_signals, axis=0))**2
    
    # Find the component most correlated with the green channel's spectrum
    best_component_idx = -1
    max_correlation = -1
    
    min_hz = MIN_BPM / 60.0
    max_hz = MAX_BPM / 60.0
    valid_indices = np.where((freqs >= min_hz) & (freqs <= max_hz))
    
    for i in range(source_signals.shape[1]):
        # Compare the shape of the power spectra in the valid frequency range
        correlation = np.corrcoef(
            green_power_spectrum[valid_indices], 
            ica_power_spectra[valid_indices, i]
        )[0, 1]
        
        print(f"Component {i+1} correlation with Green channel: {correlation:.4f}")
        
        if correlation > max_correlation:
            max_correlation = correlation
            best_component_idx = i

    print(f"==> Selected Component {best_component_idx + 1} as most likely pulse signal.")

    # --- Step 6: Bandpass Filter ONLY the selected component ---
    print(f"Step 6: Applying bandpass filter to selected component ({best_component_idx + 1})...")
    
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return filtfilt(b, a, data, axis=0)

    # We filter the chosen raw source signal
    filtered_pulse_signal = bandpass_filter(source_signals[:, best_component_idx], min_hz, max_hz, fps)
    
    # --- Step 7: Final Heart Rate Extraction ---
    print("Step 7: Calculating power spectrum and extracting final heart rate...")
    final_power_spectrum = np.abs(np.fft.fft(filtered_pulse_signal))**2
    
    # Find the peak in the valid frequency range
    peak_index_in_cropped = np.argmax(final_power_spectrum[valid_indices])
    # Get the frequency corresponding to that peak
    dominant_freq_hz = freqs[valid_indices][peak_index_in_cropped]
    
    final_heart_rate = dominant_freq_hz * 60.0

    print("\n--- Results ---")
    print(f"==> Final Estimated Heart Rate: {final_heart_rate:.2f} BPM")
    
    # --- Plotting for Debug ---
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, final_power_spectrum, label=f'Chosen Component ({best_component_idx+1}) Power')
    plt.title("Power Spectrum of Final Chosen Signal")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.xlim([min_hz, max_hz])
    plt.axvline(x=dominant_freq_hz, color='r', linestyle='--', label=f'Detected Peak ({final_heart_rate:.2f} BPM)')
    plt.axvline(x=80/60, color='k', linestyle=':', label='Actual HR (80 BPM)')
    plt.legend()
    plt.grid(True)
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
                 raise FileNotFoundError(f"Numpy file not found: {input_path}")
            video_data = np.load(input_path)
            fps = DEFAULT_FPS
            print(f"Using default FPS for .npy file: {fps} Hz")
        else:
            print(f"Processing video file: {input_path}")
            video_data, fps = process_video_to_roi_array(input_path)

    except (IndexError, FileNotFoundError) as e:
        print(f"Info: No valid video file provided or file not found. {e}")
        print(f"Attempting to load default numpy file: {DEFAULT_VIDEO}")
        try:
            video_data = np.load(DEFAULT_VIDEO)
            fps = DEFAULT_FPS
            print(f"Successfully loaded {DEFAULT_VIDEO}. Using default FPS: {fps} Hz")
        except FileNotFoundError:
            print(f"Error: Default file {DEFAULT_VIDEO} not found. Please provide a video file path or place 'data1.npy' in the script's directory.")
            sys.exit(1)

    if video_data is not None and video_data.size > 0:
        print("\n--- Video Data Info ---")
        print(f"Shape: {video_data.shape}")
        print(f"Data Type: {video_data.dtype}")
        analyze_pulse(video_data, fps)
    else:
        print("Could not load or process video data. Exiting.")