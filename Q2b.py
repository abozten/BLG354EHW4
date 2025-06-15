#Arda Barış Özten - 820220303
#Berra Mutlu - 820220331
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import hamming
from sklearn.cluster import DBSCAN



def radar(x, fs, T_0, g, T_out, T_ref, fc, r, a, v=None):
    # Constants
    c = 0.3  # km/us
    if v is None:
        v = np.zeros(len(r))

    # Ensure arrays are column vectors
    r = np.array(r).reshape(-1, 1)
    a = np.array(a).reshape(-1, 1)
    g = np.array(g).reshape(1, -1)
    T_0 = np.array(T_0).reshape(1, -1)

    x = np.array(x).flatten()
    Mx = len(x)
    delta_t = 1 / fs  # in microseconds
    T_p = Mx * delta_t
    t_x = delta_t * np.arange(Mx)

    # Fit quadratic phase to chirp
    x_ph = np.unwrap(np.angle(x))
    q = np.polyfit(t_x, x_ph, 2)
    xfit = np.polyval(q, t_x)
    corr_coef = np.dot(x_ph, xfit) / (np.linalg.norm(x_ph) * np.linalg.norm(xfit))
    if corr_coef < 0.99:
        raise ValueError("No quadratic phase match!")

    # Output time vector (ensure correct size)
    t_y = np.arange(T_out[0], T_out[1] + delta_t, delta_t)
    Mr = len(t_y)  # should be 501
    Nr = g.shape[1]
    y = np.zeros((Mr, Nr), dtype=complex)

    for i in range(len(r)):
        ri = r[i, 0]
        vi = v[i] / 1e9  # Convert m/s to km/us
        for j in range(Nr):
            r_at_T0 = ri - vi * T_0[0, j]
            tau = r_at_T0 / (c / 2 + vi)
            tmax = tau + T_p
            if tau >= T_out[1] or tmax <= T_out[0]:
                continue
            t_in_pulse = t_y - tau
            n_out = np.where((t_in_pulse >= 0) & (t_in_pulse < T_p))[0]
            if len(n_out) < 1:
                continue
            doppler_phase = np.exp(1j * 2 * np.pi * 2 * fc * (-r_at_T0 + vi * t_y[n_out]) / c)
            chirp_phase = np.exp(1j * np.polyval(q, t_in_pulse[n_out]))
            y[n_out, j] += a[i, 0] * g[0, j] * doppler_phase * chirp_phase

    return y

def dchirp(TW, p):
    N = int(p * TW)
    alpha = TW / (2 * N**2)
    n = np.arange(N)
    s = np.exp(1j * 2 * np.pi * alpha * (n - N/2)**2)
    return s
	
# --- Radar parameters ---	
fc = 10e3         # MHz (10 GHz)
fs = 10           # MHz
T = 10            # µs
W = 5             # MHz
Nrange = 501      # Sample number in range direction
Npulses = 12      # Pulse number
PRI = 200         # µs
c = 0.3           # km/µs

# --- Chirp waveform (equivalent to dchirp) ---
# to do create chirp pulse s
TW = T * W
p = fs / W
s = dchirp(TW, p)


# --- Target parameters --- number of target parameter in each array must MATCH with each other
target_range = np.array([17, 20])   # km
target_velocity = np.array([0, 0])  # m/s for simplicity targets are stationary
target_amp = np.array([1, 1])         # unit amplitude

# --- Pulse and output timing
T_0 = np.arange(Npulses) * PRI     # µs
g = np.ones(Npulses)
T_out = [100, 150]  # µs
T_ref = 0

# --- Simulate radar return ---
y = radar(s, fs, T_0, g, T_out, T_ref, fc, target_range, target_amp, target_velocity)
print(f"Shape of simulated return y: {y.shape}") # it should be matrix y[Nrange, Npulses]

# --- Matched Filtering and Plotting (as per homework PDF description) ---
# 1. Create the matched filter (time-reversed conjugate of the pulse)
matched_filter = np.conj(np.flip(s))

# 2. Convolve each pulse return (column of y) with the matched filter
x_conv = np.zeros_like(y, dtype=complex)
for i in range(Npulses):
    x_conv[:, i] = convolve(y[:, i], matched_filter, mode='same')

# 3. Create the range axis for plotting using the formula from the homework PDF
# Units: c in km/us, fs in MHz, T in us, T_out in us. Result is in km.
R_spec = (c / 2) * (np.arange(Nrange) / fs - T / 2 + T_out[0])

# 4. Plot the magnitude of the matched filter output for the first pulse
plt.figure(figsize=(12, 6))
plt.plot(R_spec, np.abs(x_conv[:, 0])) # Plotting the first pulse return
plt.title('Matched Filter Output vs. Range (First Pulse)')
plt.xlabel('Range (km)')
plt.ylabel('Magnitude')
plt.grid(True)
# Mark the expected target locations for verification
plt.axvline(x=target_range[0], color='r', linestyle='--', label=f'Target 1 @ {target_range[0]} km')
plt.axvline(x=target_range[1], color='g', linestyle='--', label=f'Target 2 @ {target_range[1]} km')
plt.legend()
plt.tight_layout()
plt.savefig("q2b_matched_filter_output_first_pulse.png")  # Save figure

# 5. Plot the 2D range-pulse map after matched filtering
plt.figure(figsize=(10, 7))
plt.imshow(np.abs(x_conv), aspect='auto', origin='lower',
           extent=[0, Npulses-1, R_spec[0], R_spec[-1]])
plt.title('Range vs. Pulse Number (After Matched Filtering)')
plt.xlabel('Pulse Number')
plt.ylabel('Range (km)')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.savefig("q2b_range_vs_pulse_matched_filtering.png")  # Save figure

plt.show()

'''
# Q2c.py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import hamming
from sklearn.cluster import DBSCAN
import os
import argparse # Add this import

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Radar Target Detection Script")
    parser.add_argument('--thresh', type=float, help='Detection threshold value.')
    args = parser.parse_args()

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

    # Determine threshold
    if args.thresh is not None:
        detection_threshold = args.thresh
        print(f"Using user-provided threshold: {detection_threshold}")
    else:
        # Dynamically set threshold, e.g., 5 times the mean magnitude
        noise_level_estimate = np.mean(np.abs(RDmapw))
        detection_threshold = 5 * noise_level_estimate 
        print(f"Using dynamically calculated threshold: {detection_threshold:.2f} (5 * mean magnitude)")

    # you can make the detection automaticly by setting a threshold to choose targets above the threshold
    targets = detect_targets_clustered(RDmapw, R_spec, vel_axis, thresh=detection_threshold)

    # Print detection results
    if targets:
        print("\\n--- Detected Targets ---")
        for i, (r_m, v_m, mag) in enumerate(targets):
            print(f"Target {i+1}: Range = {r_m:.2f} km, Velocity = {v_m:.2f} m/s, Peak Magnitude = {mag:.2f}")
    else:
        print("\\nNo targets were detected with the current threshold.")

    # Plot the Range-Doppler Map with detections marked
    plot_doppler_range_map(RDmapw, R_spec, vel_axis, targets)

    # Save the plot to a file
    plt.savefig("q2c_RDmapw_with_detections.png")

    plt.show()

'''