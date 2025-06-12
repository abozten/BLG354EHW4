# Q2c.py
#Arda Barış Özten - 820220303
#Berra Mutlu - 820220331
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import hamming
from sklearn.cluster import DBSCAN
import os

def dchirp(TW, p):
    N = int(p * TW)
    if N == 0:
        raise ValueError("Cannot create a chirp with zero length. Check T, W, p parameters.")
    alpha = TW / (2 * N**2)
    n = np.arange(N)
    s = np.exp(1j * 2 * np.pi * alpha * (n - N/2)**2)
    return s

def detect_targets_clustered(rd_map, range_axis, vel_axis, thresh, eps=2, min_samples=1):
    mags = np.abs(rd_map)
    detections = np.argwhere(mags > thresh)
    if len(detections) == 0:
        print(f"No detections found above threshold={thresh}. You might need to lower it.")
        return []

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(detections)
    labels = clustering.labels_
    unique_labels = set(labels) - {-1}

    targets = []
    for label in unique_labels:
        points = detections[labels == label]
        mags_cluster = [mags[pt[0], pt[1]] for pt in points]
        peak_idx = np.argmax(mags_cluster)
        r_idx, v_idx = points[peak_idx]
        r_m = range_axis[r_idx]
        v_m = vel_axis[v_idx]
        targets.append((r_m, v_m, mags[r_idx, v_idx]))

    return targets

def plot_doppler_range_map(rd_map, range_axis, vel_axis, targets=None):
    plt.figure(figsize=(10, 6))
    plt.imshow(np.abs(rd_map), aspect='auto', extent=[vel_axis[0], vel_axis[-1], range_axis[-1], range_axis[0]])
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Range (km)')
    plt.title('Doppler-Range Map with Target Detections')
    plt.colorbar(label='Magnitude')
    if targets:
        for (r_m, v_m, _) in targets:
            plt.plot(v_m, r_m, 'rx', markersize=8, markeredgewidth=2)
    plt.tight_layout()

#to do you will have the same functions and steps with Q2b.py
# 1. Define parameters based on radar_simulation.pdf Table 10.2 for rad100.npy
fc = 7000         # MHz (7 GHz)
T = 7             # µs
W = 7             # MHz
fs = 8            # MHz
PRI = 60          # µs
T_out_start = 25  # µs (start of receive window from Table 10.2)
c_kms = 0.3       # Speed of light in km/µs
c_ms = 3e8        # Speed of light in m/s

# 2. Load the radar data file rad100.npy
DATA_FILE = 'rad100.npy'
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(f"Error: Data file '{DATA_FILE}' not found. Please place it in the same directory as the script.")
y = np.load(DATA_FILE)
Nrange, Npulses = y.shape
print(f"Loaded data from '{DATA_FILE}', shape: ({Nrange}, {Npulses})")

# 3. Synthesize the transmitted chirp pulse `s` for the matched filter
TW = T * W
p = fs / W
s = dchirp(TW, p)

# 4. Perform matched filtering on the loaded data to get `x_conv`
matched_filter = np.conj(np.flip(s))
x_conv = np.zeros_like(y, dtype=complex)
for i in range(Npulses):
    x_conv[:, i] = convolve(y[:, i], matched_filter, mode='same')

# The following processing steps are from the provided Q2c.py snippet
# apply a hamming windowing (this will reduce the target amplitude and effects on the neighbour range bins)
ham2d = np.outer(hamming(Nrange), hamming(Npulses))
x_conv_win = x_conv * ham2d
# take the fft in row (azimuth direction ) we will find the range velocities
# azimuth resolution was very low 12 pulses now we pad zeros to increase the number by 2*Nrange so that we can estimate velocities better
RDmapw = np.fft.fftshift(np.fft.fft(x_conv_win, n=2*Nrange, axis=1), axes=1)

# --- Define Range and Velocity Axes (needed for plotting and detection) ---
# Range axis calculation
# time = start_time + sample_index / sampling_freq
# range = speed_of_light * time / 2
time_axis_us = T_out_start + np.arange(Nrange) / fs
R_spec = c_kms * time_axis_us / 2  # Range in km

# Velocity axis calculation
lambda_radar = c_ms / (fc * 1e6)                # Wavelength in meters (fc in MHz -> Hz)
vel_max = lambda_radar / (4 * PRI * 1e-6)    # Max unambiguous velocity in m/s
vel_axis = np.linspace(-vel_max, vel_max, 2*Nrange) # FFT length is 2*Nrange

# --- Plotting and Detection ---
# Plot the raw Range-Doppler Map
plt.figure(figsize=(10, 6))
plt.imshow(np.abs(RDmapw), aspect='auto', extent=[vel_axis[0], vel_axis[-1], R_spec[-1], R_spec[0]])
plt.xlabel('Velocity (m/s)')
plt.ylabel('Range (km)')
plt.title('Matched Filtered Range-Doppler Map')
plt.colorbar(label='Magnitude')
plt.tight_layout()

# you can make the detection automaticly by setting a threshold to choose targets above the threshold
targets = detect_targets_clustered(RDmapw, R_spec, vel_axis, thresh=90)

# Print detection results
if targets:
    print("\n--- Detected Targets ---")
    for i, (r_m, v_m, mag) in enumerate(targets):
        print(f"Target {i+1}: Range = {r_m:.2f} km, Velocity = {v_m:.2f} m/s, Peak Magnitude = {mag:.2f}")
else:
    print("\nNo targets were detected with the current threshold.")

# Plot the Range-Doppler Map with detections marked
plot_doppler_range_map(RDmapw, R_spec, vel_axis, targets)

# Save the plot to a file
plt.savefig("q2c_RDmapw_with_detections.png")

plt.show()

