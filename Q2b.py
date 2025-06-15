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

