#Arda Barış Özten - 820220303
#Berra Mutlu - 820220331
import numpy as np
import matplotlib.pyplot as plt

def dchirp(TW, p):
    N = int(p * TW)
    alpha = TW / (2 * N**2)
    n = np.arange(N)
    s = np.exp(1j * 2 * np.pi * alpha * (n - N/2)**2)
    return s

def plotspec(x, Ts):
	#TODO :
    # calculate the dft and frequency spectrum 
    # remember there will be negative frequencies as well

    # Use zero-padding for a smoother plot. A power of 2 is efficient for FFT.
    N_fft = 4096 

    # Calculate the DFT.
    fxs_full = np.fft.fft(x, n=N_fft)
    
    # Calculate the frequency axis. fftfreq generates frequencies for the un-shifted FFT output.
    freqs_full = np.fft.fftfreq(N_fft, d=Ts)
    
    # Shift the zero-frequency component to the center for plotting.
    fxs = np.fft.fftshift(fxs_full)
    freqs = np.fft.fftshift(freqs_full)


    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(fxs))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude (linear units)')
    plt.title('Spectrum of Chirp Signal')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.unwrap(np.angle(fxs)))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('phase (rad)')
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("q2a_chirp_spectrum.png")  # Save the spectrum plot

    return fxs, freqs
    
# Add parameters from the homework description PDF
T_param = 25e-6   # Pulse length in seconds
W_param = 2e6     # Swept bandwidth in Hz
fs_param = 20e6   # Sampling frequency in Hz
TW = T_param * W_param
p = fs_param / W_param

# to do according to the explanations
# 1. Generate the chirp signal
sofn = dchirp(TW, p)
N = len(sofn)
n = np.arange(N)

# 2. Plot the real part of the generated chirp signal
plt.figure()
plt.plot(n, np.real(sofn))
plt.title('Real part of sofn')
plt.xlabel('n (sample index)')
plt.ylabel('Real Part')
plt.grid(True)
plt.savefig("q2a_chirp_real_part.png")  # Save the real part plot

# 3. Compute and plot the spectrum of the chirp signal
Ts = 1 / fs_param
plotspec(sofn, Ts)

# 4. Display all generated plots
plt.show()

