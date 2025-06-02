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

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(freqs, np.abs(fxs))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('magnitude (linear units)')

    plt.subplot(2, 1, 2)
    plt.plot(freqs, np.unwrap(np.angle(fxs)))
    plt.xlabel('frequency (Hz)')
    plt.ylabel('phase (rad)')
    plt.tight_layout()

    return fxs, freqs
    
# Add parameters

plt.figure()
plt.plot(n, np.real(sofn))
plt.title('Real part of sofn')
plt.xlabel('n')

# to do according to the explanations
    
