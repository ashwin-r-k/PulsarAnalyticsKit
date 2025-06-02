

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from scipy.signal import welch, correlate, find_peaks
from scipy import stats
from scipy.stats import linregress
from scipy.optimize import curve_fit, minimize_scalar


def compute_channel_intensity_matrix(channel_data,block_size, avg_blocks, sample_rate):
    total_samples = len(channel_data)
    segment_length = block_size * avg_blocks
    num_segments = total_samples // segment_length
    trimmed = channel_data[:num_segments * segment_length]
    reshaped = trimmed.reshape(num_segments * avg_blocks, block_size)
    fft_data = np.fft.rfft(reshaped, axis=1)  # Use rfft for real inputs
    power_spectra = np.abs(fft_data) ** 2
    power_avg = power_spectra.reshape(num_segments, avg_blocks, -1).mean(axis=1)
    return power_avg

# computing De dispersion

def dedisperse(matrix, DM,block_size, avg_blocks , sample_rate , bandwidth_MHZ = 16.5,center_freq_MHZ = 326.5):

    """
    Dedisperse intensity matrix using cold plasma dispersion delay.
    DM is in pc/cm^3 should be DM=67.8 for vela
    """

    n_time, n_freq = matrix.shape
    #print(matrix.shape)

    # # Generate frequency array for each channel
    bandwidth = bandwidth_MHZ /1000 # MHz to GHz
    center_freq = center_freq_MHZ / 1000 # MHz to GHz

    freq_array = np.linspace(center_freq - bandwidth / 2, center_freq + bandwidth / 2, n_freq)


    # Reference frequency (earliest arrival): highest frequency
    f_ref = freq_array[len(freq_array)//2]

    # Calculate delay in microseconds for each frequency channel
    delays_ms = 4.15 * DM * (1 / freq_array**2 - 1 / f_ref**2)  # in ms
    #delays_s = delays_ms  / 1000  # to Sec


    # # Time bin duration (microseconds per row)
    t_bin =  avg_blocks * block_size / sample_rate * 1000 # in mili Sec
    #print( "Time bin",t_bin)

    delay_bins = (delays_ms / t_bin).astype(int)
    
    # # Initialize dedispersed matrix
    dedispersed = np.zeros_like(matrix)

    for i in range(n_freq):
        shift = delay_bins[i]
        dedispersed[:, i] = np.roll(matrix[:, i], shift)
        if shift > 0 :
            dedispersed[:abs(shift),i] = 0
        elif shift < 0:
            dedispersed[shift:,i] = 0

    return dedispersed


def sharpness_score(matrix,pulseperiod,t_bin_ms,NPeeks,to_plot):
    profile = matrix.sum(axis=1)
    distance = int(pulseperiod / t_bin_ms  * 0.7 )
    width=( int(pulseperiod / t_bin_ms * 0.1) , int(pulseperiod / t_bin_ms * 0.5 ))
    print(distance,width)
    print(len(profile))
    return -fit_multiple_gaussians(profile,num_peaks = NPeeks ,distance=distance,width=width,to_plot=to_plot)  # Lower sigma → better alignment

def find_best_dm(matrix, center_freq_MHZ,bandwidth_MHZ, sample_rate, block_size, avg_blocks,Npeaks,pulseperiod_ms, to_plot, dm_min=0, dm_max=100, tol=1):

    def objective(dm):
        t_bin_ms = (block_size * avg_blocks / sample_rate) * 1e3
        dedispersed = dedisperse(matrix, dm ,block_size, avg_blocks , sample_rate , center_freq_MHZ, bandwidth_MHZ)
        score = sharpness_score(dedispersed ,pulseperiod_ms,t_bin_ms, Npeaks,to_plot)

        print(f"DM = {dm}  ; score = {score}")
        #plot_intensity_matrix(dedisperse_matrix(matrix, dm, block_size, avg_blocks , sample_rate), block_size, avg_blocks , sample_rate,gamma=2.5)

        return score

    result = minimize_scalar(objective, bounds=(dm_min, dm_max), method='bounded', options={'xatol': tol})

    return result.x, -result.fun


def gaussian(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def fit_multiple_gaussians(profile, num_peaks, distance, width,to_plot):
    peaks, _ = find_peaks(profile, distance=distance, width=width)
    print(width)
    peaks = peaks[:num_peaks]
    sigmas = []

    if to_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(profile, label='Profile')
    x_full = np.arange(len(profile))

    for peak in peaks:
        try:
            x = np.arange(peak - 30, peak + 30)
            x = x[(x >= 0) & (x < len(profile))]
            y = profile[x]
            p0 = [np.max(y), peak, 10, np.median(profile)]
            popt, _ = curve_fit(gaussian, x, y, p0=p0)
            sigmas.append(abs(popt[2]))
            if to_plot:
                plt.plot(x, gaussian(x, *popt), '--', label=f'Gaussian fit (peak {peak})')
        except Exception as e:
            continue
    if to_plot:
        plt.scatter(peaks, profile[peaks], color='red', zorder=5, label='Peaks')
        plt.legend()
        plt.title('Gaussian Fits to Profile Peaks')
        plt.xlabel('Sample')
        plt.ylabel('Intensity')
        plt.tight_layout()
        plt.show()
    return np.mean(sigmas) if sigmas else np.inf


