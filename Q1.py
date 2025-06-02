import numpy as np
import matplotlib.pyplot as plt
from scipy import cluster, signal
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
    """
    Compute the 1D single-sided DFT (magnitude) of a real signal,
    then normalizing to one.
    """
    X = np.abs(np.fft.rfft(x))
    return X/X.max()		

fs, stereo = wavfile.read('adam_curtis.wav')
print('Original shape:', stereo.shape)
print('Sampling frequency:', fs)
# TODO we don't need the stereo so by using np.mean you 
# can average the two channels and convert it to mono
mono = mono.astype(np.float32)/np.iinfo(np.int16).max
sd.default.samplerate = fs
sd.default.channels = 1
play(mono)


fr1, h_lpf = signal.freqz(lpf, fs=fs)
h_mono = normalized_spectrum(mono)
h_mod = normalized_spectrum(modulated)
fr_n = np.linspace(0, 0.5*fs, h_mono.size)
plt.figure(figsize=(10, 6))
plt.plot(fr1, np.abs(h_lpf), color='tab:red', linestyle='dashed')
plt.plot(fr_n, h_mono, label='Original', color='tab:blue', alpha=0.3)
plt.plot(fr_n, h_mod, label='Amplitude-Modulated', color='tab:orange', alpha=0.3)
plt.legend()

np.round(np.mean((demodulated - mono)**2), 5)

#TODO:
#you have to define following function:
def amodulate()
def ademodulate()


