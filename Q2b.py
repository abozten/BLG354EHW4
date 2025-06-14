#Arda Barış Özten - 820220303
#Berra Mutlu - 820220331
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve

# ==============================================================================
# Part 1: Single Target at 1500m (The Missing Part)
#
# This section implements the requirement to simulate a single target,
# perform matched filtering via both time-domain convolution and
# frequency-domain multiplication, and plot the results.
# ==============================================================================

def dchirp(TW, p):
    """Generates a discrete chirp signal."""
    N = int(p * TW)
    alpha = TW / (2 * N**2)
    n = np.arange(N)
    s = np.exp(1j * 2 * np.pi * alpha * (n - N/2)**2)
    return s

# --- Parameters from Q2a for this specific task ---
T_param = 25e-6   # Pulse length in seconds (25 us)
W_param = 2e6     # Swept bandwidth in Hz (2 MHz)
fs_param = 20e6   # Sampling frequency in Hz (20 MHz)
TW = T_param * W_param
p = fs_param / W_param
s_chirp = dchirp(TW, p)

# --- Matched Filter ---
matched_filter_h = np.conj(np.flip(s_chirp))

# --- Target Simulation ---
target_range_m = 1500  # Target at 1500m
c = 3e8                # Speed of light in m/s
Td = 2 * target_range_m / c  # Round-trip time delay
delay_samples = int(np.round(Td * fs_param))

# Create a receive buffer and place the delayed chirp in it
buffer_length = len(s_chirp) + delay_samples
y_received = np.zeros(buffer_length, dtype=complex)
y_received[delay_samples:delay_samples + len(s_chirp)] = s_chirp

# --- Time-Domain Convolution ---
x_conv_time = convolve(y_received, matched_filter_h, mode='same')

# --- Frequency-Domain Multiplication ---
# This demonstrates the Convolution Theorem: conv(a, b) <=> FFT(a) * FFT(b)
# The FFT of a matched filter h=conj(flip(s)) is conj(FFT(s))
S_fft = np.fft.fft(s_chirp, n=buffer_length)
Y_fft = np.fft.fft(y_received, n=buffer_length)
X_mult_freq = Y_fft * np.conj(S_fft)
x_mult_time = np.fft.ifft(X_mult_freq)

# --- Plotting the Results for the Single Target ---
time_axis = np.arange(buffer_length) / fs_param * 1e6 # Time in microseconds

fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig1.suptitle('Matched Filter Demonstration for a Single Target at 1500m', fontsize=16)

# Plot Time-Domain Convolution Result
ax1.plot(time_axis, np.abs(x_conv_time))
ax1.axvline(x=Td * 1e6, color='r', linestyle='--', label=f'Expected Delay Td = {Td*1e6:.2f} µs')
ax1.set_title('Time-Domain Convolution')
ax1.set_xlabel('Time (µs)')
ax1.set_ylabel('Magnitude')
ax1.legend()
ax1.grid(True)

# Plot Frequency-Domain Multiplication Result
ax2.plot(time_axis, np.abs(x_mult_time))
ax2.axvline(x=Td * 1e6, color='r', linestyle='--', label=f'Expected Delay Td = {Td*1e6:.2f} µs')
ax2.set_title('Frequency-Domain Multiplication (via IFFT)')
ax2.set_xlabel('Time (µs)')
ax2.set_ylabel('Magnitude')
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("q2b_single_target_1500m.png")


# ==============================================================================
# Part 2: Multiple Target Simulation (Original Q2b Code)
#
# This section simulates the radar return from two targets at 17 and 20 km
# and plots the resulting range spectrum after matched filtering.
# ==============================================================================

def radar(x, fs, T_0, g, T_out, T_ref, fc, r, a, v=None):
    """Simulates radar returns from multiple targets."""
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

    # Output time vector
    t_y = np.arange(T_out[0], T_out[1] + delta_t, delta_t)
    Mr = len(t_y)
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

# --- Radar parameters for multi-target simulation ---
fc = 10e3         # MHz (10 GHz)
fs = 10           # MHz
T = 10            # µs
W = 5             # MHz
Nrange = 501      # Sample number in range direction
Npulses = 12      # Pulse number
PRI = 200         # µs
c_kms = 0.3       # km/µs

# --- Chirp waveform for this scenario ---
TW_multi = T * W
p_multi = fs / W
s_multi = dchirp(TW_multi, p_multi)

# --- Target parameters ---
target_range = np.array([17, 20])   # km
target_velocity = np.array([0, 0])  # m/s
target_amp = np.array([1, 1])

# --- Pulse and output timing ---
T_0 = np.arange(Npulses) * PRI
g = np.ones(Npulses)
T_out = [100, 150]
T_ref = 0

# --- Simulate radar return ---
y_multi = radar(s_multi, fs, T_0, g, T_out, T_ref, fc, target_range, target_amp, target_velocity)

# --- Matched Filtering for multi-target scenario ---
matched_filter_multi = np.conj(np.flip(s_multi))
x_conv_multi = np.zeros_like(y_multi, dtype=complex)
for i in range(Npulses):
    x_conv_multi[:, i] = convolve(y_multi[:, i], matched_filter_multi, mode='same')

# --- Create the range axis for plotting ---
R_spec = (c_kms / 2) * (np.arange(Nrange) / fs - T / 2 + T_out[0])

# --- Plot the magnitude of the matched filter output for the first pulse ---
plt.figure(figsize=(12, 6))
plt.plot(R_spec, np.abs(x_conv_multi[:, 0]))
plt.title('Matched Filter Output vs. Range (Two Targets at 17km & 20km)')
plt.xlabel('Range (km)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.axvline(x=target_range[0], color='r', linestyle='--', label=f'Target 1 @ {target_range[0]} km')
plt.axvline(x=target_range[1], color='g', linestyle='--', label=f'Target 2 @ {target_range[1]} km')
plt.legend()
plt.tight_layout()
plt.savefig("q2b_matched_filter_output_multi_target.png")

# --- Display all generated plots ---
plt.show()