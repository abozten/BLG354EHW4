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
T = 10           # µs
W = 5             # MHz
Nrange = 501      # Sample number in range direction
Npulses = 12      # Pulse number
PRI = 200         # µs
c = 0.3           # km/µs

# --- Chirp waveform (equivalent to dchirp) ---

# to do create chirp pulse s


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
print(y.shape) # it should be matrix y[Nrange, Npulses]
