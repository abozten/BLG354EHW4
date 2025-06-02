import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
from scipy.signal.windows import hamming
from sklearn.cluster import DBSCAN

def detect_targets_clustered(rd_map, range_axis, vel_axis, thresh, eps=2, min_samples=1):
    mags = np.abs(rd_map)
    detections = np.argwhere(mags > thresh)
    if len(detections) == 0:
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
    plt.show()

#to do you will have the same functions and steps with Q2b.py
#once you have the convolution with match filter result let say x_conv

# apply a hamming windowing (this will reduce the target amplitude and effects on the neighbour range bins)
ham2d = np.outer(hamming(Nrange), hamming(Npulses))
x_conv_win = x_conv * ham2d
# take the fft in row (azimuth direction ) we will find the range velocities
# azimuth resolution was very low 12 pulses now we pad zeros to increase the number by 2*Nrange so that we can estimate velocities better
RDmapw = np.fft.fftshift(np.fft.fft(x_conv_win, n=2*Nrange, axis=1), axes=1)
# targets are moving only in range directions towards to sensor positive, away from the sensor negative values
# there is maximum target velocity limit that we can estimate which is related with the PRI and the fc (radar wavelenth)
# max doppler frequency shift (fd) (detected) is equal to fd_max = = PRF/2 = 1/(PRI*2)
# from the radar_simulation.pdf file you can check the target velocity processing 
# target velocity cause a doppler shift it is equal to fd = 2*vt/Lamda_radar   
lambda_radar = c*1e3 / fc                # m
vel_max = lambda_radar / (4 * PRI * 1e-6)    # m/s from the fd_max = PRF/2
vel_axis = np.linspace(-vel_max, vel_max, 2*Nrange)

plt.imshow(np.abs(RDmapw), aspect='auto', extent=[vel_axis[0], vel_axis[-1], R_spec[-1], R_spec[0]])
plt.xlabel('Velocity (m/s)')
plt.ylabel('Range (km)')
plt.title('Matched Filtered Range-Doppler Map')
plt.colorbar(label='Magnitude')
plt.tight_layout()
plt.show()

# you can make the detection automaticly by setting a threshold to choose targets above the threshold
targets = detect_targets_clustered(RDmapw, R_spec, vel_axis, thresh=90)

for i, (r_m, v_m, mag) in enumerate(targets):
    print(f"Target {i+1}: Range = {r_m:.2f} km, Velocity = {v_m:.2f} m/s, Peak = {mag:.2f}")
plot_doppler_range_map(RDmapw, R_spec, vel_axis, targets)