import numpy as np
import matplotlib.pyplot as plt
import os

# Create directory for saving plots
if not os.path.exists('radar_plots'):
    os.makedirs('radar_plots')

# Radar parameters from the table
T = 25e-6  # Pulse length in seconds (25 μs)
W = 2e6    # Swept bandwidth in Hz (9 MHz)
fs = 20e6  # Sampling frequency in Hz (20 MHz)
TW = 50    # Time-bandwidth product (dimensionless)
p_ratio = 10  # Oversampling ratio (dimensionless)

# Calculate p = fs/W
p = fs / W
print(f"Calculated p = fs/W = {p:.2f}")
print(f"Given oversampling ratio = {p_ratio}")

def dchirp(TW, p):
    """Generate discrete chirp signal"""
    N = int(p * TW)
    alpha = TW / (2 * N**2)
    n = np.arange(N)
    s = np.exp(1j * 2 * np.pi * alpha * (n - N/2)**2)
    return s

def plotspec(x, Ts):
    """Plot frequency spectrum of signal"""
    N = len(x)
    fxs = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, Ts)
    
    # Sort frequencies for better visualization
    idx = np.argsort(freqs)
    freqs = freqs[idx]
    fxs = fxs[idx]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(fxs))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude (linear units)')
    plt.title('Magnitude Spectrum')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.unwrap(np.angle(fxs)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Phase (rad)')
    plt.title('Phase Spectrum')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('radar_plots/q2a_chirp_frequency_spectrum.png', dpi=300, bbox_inches='tight')
    
    return fxs, freqs

# Step 1: Generate chirp signal
print(f"\nGenerating chirp signal with TW={TW}, p={p}")
s = dchirp(TW, p)
N = len(s)
print(f"Chirp signal length N = {N} samples")

# Time vector
Ts = 1/fs  # Sampling period
n = np.arange(N)
t = n * Ts

# Step 2: Plot real part of chirp signal in time domain
plt.figure(figsize=(12, 6))
plt.plot(n, np.real(s))
plt.title('Real part of Chirp Signal (Time Domain)')
plt.xlabel('Sample index n')
plt.ylabel('Amplitude')
plt.grid(True)
plt.savefig('radar_plots/q2a_chirp_real_time_domain.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 3: Plot chirp signal in frequency domain
print("\nPlotting chirp signal frequency spectrum...")
fxs, freqs = plotspec(s, Ts)

# Step 4: Create matched filter
print("\nCreating matched filter...")
matched_filter = np.conj(np.flip(s))

# Step 5: Convolution in time domain (chirp with its matched filter)
print("Performing convolution in time domain...")
conv_result = np.convolve(s, matched_filter, mode='same')

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(conv_result)), np.abs(conv_result))
plt.title('Matched Filter Output (Time Domain Convolution)')
plt.xlabel('Sample index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.savefig('radar_plots/q2a_matched_filter_time_convolution.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 6: Multiplication in frequency domain
print("Performing multiplication in frequency domain...")
S_freq = np.fft.fft(s, N)
MF_freq = np.fft.fft(matched_filter, N)
mult_result_freq = S_freq * MF_freq
mult_result_time = np.fft.ifft(mult_result_freq)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(mult_result_time)), np.abs(mult_result_time))
plt.title('Matched Filter Output (Frequency Domain Multiplication)')
plt.xlabel('Sample index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.savefig('radar_plots/q2a_matched_filter_freq_multiplication.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 7: Target at 1500m range
print(f"\nCalculating delayed signal for target at 1500m...")
c = 3e8  # Speed of light in m/s
target_range = 1500  # meters
Td = 2 * target_range / c  # Two-way delay time
print(f"Delay time Td = {Td*1e6:.2f} μs")

# Calculate delay in samples
delay_samples = int(Td / Ts)
print(f"Delay in samples = {delay_samples}")

# Create delayed chirp signal (zero-padded)
total_length = N + delay_samples
s_delayed = np.zeros(total_length, dtype=complex)
s_delayed[delay_samples:delay_samples+N] = s

# Extend matched filter to same length
matched_filter_extended = np.zeros(total_length, dtype=complex)
matched_filter_extended[:N] = matched_filter

# Step 8: Convolution with delayed signal
print("Convolving delayed signal with matched filter...")
conv_delayed = np.convolve(s_delayed, matched_filter, mode='same')

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(conv_delayed)), np.abs(conv_delayed))
plt.title('Matched Filter Output with Delayed Target Signal (Time Domain)')
plt.xlabel('Sample index')
plt.ylabel('Magnitude')
plt.grid(True)
# Mark the expected peak location
plt.axvline(x=len(conv_delayed)//2 + delay_samples, color='r', linestyle='--', 
           label=f'Expected peak at delay = {delay_samples} samples')
plt.legend()
plt.savefig('radar_plots/q2a_target_detection_time_domain.png', dpi=300, bbox_inches='tight')
plt.show()

# Step 9: Frequency domain multiplication with delayed signal
print("Frequency domain processing with delayed signal...")
S_delayed_freq = np.fft.fft(s_delayed)
MF_extended_freq = np.fft.fft(matched_filter_extended)
mult_delayed_freq = S_delayed_freq * MF_extended_freq
mult_delayed_time = np.fft.ifft(mult_delayed_freq)

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(mult_delayed_time)), np.abs(mult_delayed_time))
plt.title('Matched Filter Output with Delayed Target Signal (Frequency Domain)')
plt.xlabel('Sample index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.axvline(x=delay_samples, color='r', linestyle='--', 
           label=f'Expected peak at delay = {delay_samples} samples')
plt.legend()
plt.show()

# Step 10: Range resolution analysis
print(f"\nRange Resolution Analysis:")
range_resolution = c / (2 * W)
print(f"Theoretical range resolution = c/(2W) = {range_resolution:.2f} m")

# Find peak in matched filter output
peak_idx = np.argmax(np.abs(conv_delayed))
detected_delay_samples = peak_idx - len(conv_delayed)//2
detected_range = (detected_delay_samples * Ts * c) / 2
print(f"Detected peak at sample {peak_idx}")
print(f"Detected delay = {detected_delay_samples} samples")
print(f"Detected range = {detected_range:.1f} m")
print(f"Range error = {abs(detected_range - target_range):.1f} m")

# Summary plot showing all key results
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.plot(t*1e6, np.real(s))
plt.title('Original Chirp Signal (Real Part)')
plt.xlabel('Time (μs)')
plt.ylabel('Amplitude')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(freqs/1e6, np.abs(fxs))
plt.title('Chirp Signal Spectrum')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(np.arange(len(conv_result)), np.abs(conv_result))
plt.title('Matched Filter Autocorrelation')
plt.xlabel('Sample index')
plt.ylabel('Magnitude')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(np.arange(len(conv_delayed)), np.abs(conv_delayed))
plt.title('Target Detection (1500m range)')
plt.xlabel('Sample index')
plt.ylabel('Magnitude')
plt.axvline(x=peak_idx, color='r', linestyle='--', alpha=0.7)
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"\n=== SUMMARY ===")
print(f"Chirp parameters: T={T*1e6:.1f}μs, W={W/1e6:.1f}MHz, TW={TW}")
print(f"Sampling: fs={fs/1e6:.1f}MHz, N={N} samples")
print(f"Target range: {target_range}m")
print(f"Delay time: {Td*1e6:.2f}μs ({delay_samples} samples)")
print(f"Detected range: {detected_range:.1f}m")
print(f"Range resolution: {range_resolution:.2f}m")