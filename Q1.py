#Arda Barış Özten - 820220303
#Berra Mutlu - 820220331
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sounddevice as sd

if 'ggplot' in plt.style.available:
    plt.style.use('ggplot')


def play(signal):
    try:
        sd.play(signal, blocking=True)
    except KeyboardInterrupt:
        print('Interrupted playback.')


def normalize_spectrum(x):
    X = np.abs(np.fft.rfft(x))
    return X / X.max()


# Load and preprocess the audio
fs, stereo = wavfile.read('adam_curtis.wav')
print('Original shape:', stereo.shape)
print('Sampling frequency:', fs)

# Convert to mono and normalize
mono = stereo.mean(axis=1)
mono = mono.astype(np.float32) / np.iinfo(np.int16).max
sd.default.samplerate = fs
sd.default.channels = 1
play(mono)

# Carrier frequency
f0 = 8000  # 8kHz
w0 = 2 * np.pi * f0
t = np.arange(len(mono)) / fs


# Amplitude modulation
def amodulate(x, f0, fs):
    t = np.arange(len(x)) / fs
    carrier = np.cos(2 * np.pi * f0 * t)
    return x * carrier


# Amplitude demodulation (without LPF)
def ademodulate(y, f0, fs, lpf=None):
    t = np.arange(len(y)) / fs
    carrier = np.cos(2 * np.pi * f0 * t)
    demod = y * carrier * 2  # scale factor due to modulation theorem
    if lpf is not None:
        demod = signal.lfilter(lpf, [1.0], demod)
    return demod


modulated = amodulate(mono, f0, fs)
play(modulated)

# Demodulate without LPF
demodulated_raw = ademodulate(modulated, f0, fs)
play(demodulated_raw)

# Plot spectrum to choose LPF cutoff
h_mono = normalize_spectrum(mono)
h_mod = normalize_spectrum(modulated)
fr_n = np.linspace(0, 0.5 * fs, len(h_mono))

plt.figure(figsize=(10, 6))
plt.plot(fr_n, h_mono, label='Original', color='tab:blue', alpha=0.5)
plt.plot(fr_n, h_mod, label='Modulated', color='tab:orange', alpha=0.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Normalized Magnitude")
plt.title("Spectrum of Original and Modulated Signals")
plt.legend()
plt.grid(True)
plt.savefig("q1_spectrum_original_modulated.png")
plt.show()

# Design lowpass filter
cutoff = 4000  # Choose cutoff around half of carrier frequency (Nyquist of baseband)
lpf = signal.firwin(numtaps=30, cutoff=cutoff, fs=fs)

# Frequency response of LPF
fr1, h_lpf = signal.freqz(lpf, fs=fs)

# Final plot: LPF, original, modulated
plt.figure(figsize=(10, 6))
plt.plot(fr1, np.abs(h_lpf), 'r--', label='LPF')
plt.plot(fr_n, h_mono, label='Original', alpha=0.5)
plt.plot(fr_n, h_mod, label='Modulated', alpha=0.5)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.title("LPF and Signal Spectra")
plt.legend()
plt.grid(True)
plt.savefig("q1_lpf_and_signal_spectra.png")
plt.show()

# Demodulate with LPF
demodulated = ademodulate(modulated, f0, fs, lpf)
play(demodulated)

# Calculate MSE
mse = np.round(np.mean((demodulated - mono) ** 2), 5)
print("Mean Squared Error:", mse)
