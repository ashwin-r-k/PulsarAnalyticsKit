import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import  find_peaks
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

def dedisperse_pop(matrix, DM,block_size, avg_blocks , sample_rate , bandwidth_MHZ = 16.5,center_freq_MHZ = 326.5):

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
    max = delay_bins[0]
    min = delay_bins[-1]
    for i in range(n_freq):
        shift = delay_bins[i]
        dedispersed[:, i] = np.roll(matrix[:, i], shift)
        if shift > 0 :
            dedispersed[:abs(shift),i] = 0
        elif shift < 0:
            dedispersed[shift:,i] = 0
    # Chop the matrix to remove the leading and trailing zeros

    dedispersed = dedispersed[max:min,:]
    return dedispersed

def find_best_dm(matrix,center_freq_MHZ,bandwidth_MHZ, sample_rate, block_size, avg_blocks,num_peaks,pulseperiod_ms, to_plot,dm_min, dm_max, tol=1):
    t_bin_ms = (block_size * avg_blocks / sample_rate) * 1e3

    def objective(dm):
        dedispersed = dedisperse(matrix, dm ,block_size, avg_blocks , sample_rate , bandwidth_MHZ,center_freq_MHZ)
        score = sharpness_score(dedispersed,num_peaks,pulseperiod_ms,t_bin_ms, to_plot)
        print(f"DM = {dm}  ; score = {score}")
        return -1*score

    result = minimize_scalar(objective, bounds=(dm_min, dm_max), method='bounded', options={'xatol': tol})
    print(result)
    return result.x


def gaussian(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def fit_multiple_gaussians(profile, num_peaks, distance, width, to_plot=False):
    peaks, _ = find_peaks(profile, distance=distance, width=width)

    peaks = peaks[:num_peaks]
    snrs = []

    if to_plot:
        plt.figure(figsize=(10, 4))
        plt.plot(profile, label='Profile')

    width_single_gauss = int(len(profile) / (2 * num_peaks) * 0.7)

    for peak in peaks:
        x = np.arange(peak - width_single_gauss, peak + width_single_gauss)
        x = x[(x >= 0) & (x < len(profile))]
        y = profile[x]
        try:
            p0 = [np.max(y), peak, width_single_gauss // 3, np.median(profile)]
            popt, _ = curve_fit(gaussian, x, y, p0=p0, maxfev=10000)
            amp, mu, sigma, offset = popt
            # Estimate noise as std of profile excluding the fitted region
            mask = np.ones(len(profile), dtype=bool)
            mask[x] = False
            noise = np.std(profile[mask])
            snr = amp / sigma if noise > 0 else 0
            snrs.append(snr)
            if to_plot:
                plt.plot(x, gaussian(x, *popt), '--', label=f'Fit (peak {peak})')
        except Exception as e:
            print(f"Fit failed at peak {peak}: {e}")
            continue

    if to_plot:
        plt.scatter(peaks, profile[peaks], color='red', zorder=5, label='Peaks')
        plt.legend()
        plt.title('Gaussian Fits to Profile Peaks')
        plt.xlabel('Sample')
        plt.ylabel('Intensity')
        plt.tight_layout()
        plt.show()

    snrs_clean = [s for s in snrs if s is not None and np.isfinite(s)]
    
    if len(snrs_clean) >= max(1, num_peaks * 0): #max(1, num_peaks * 0.3) #ensures at least 1 peak
        return np.mean(snrs_clean)
    else:
        print(f"Warning: Only {len(snrs_clean)} valid fits (expected {num_peaks}).")
        return 0

def sharpness_score(matrix, num_peaks, pulseperiod_ms, t_bin_ms, to_plot=False):
    profile = matrix.sum(axis=1)
    distance = int(pulseperiod_ms / t_bin_ms * 0.7)
    width = (int(pulseperiod_ms / t_bin_ms * 0.1), int(pulseperiod_ms / t_bin_ms * 0.5))
    sigma_avg = fit_multiple_gaussians(profile, num_peaks=num_peaks, distance=distance, width=width, to_plot=to_plot)
    return sigma_avg if sigma_avg is not None else 0


def find_best_dm_Grid(matrix, center_freq_MHZ, bandwidth_MHZ, sample_rate, block_size, avg_blocks,
                       num_peaks, pulseperiod_ms, to_plot, dm_min, dm_max, tol):
    t_bin_ms = (block_size * avg_blocks / sample_rate) * 1e3
    scores = []
    for dm in np.linspace(dm_min, dm_max, int((abs(dm_max-dm_min) / tol) + 1)):
        dedispersed = dedisperse(matrix, dm, block_size, avg_blocks, sample_rate, bandwidth_MHZ, center_freq_MHZ)
        score = sharpness_score(dedispersed, num_peaks, pulseperiod_ms, t_bin_ms, to_plot)
        if not np.isfinite(score):
            print(f"Skipping DM = {dm} due to invalid score.")
            score = 0
        # print(f"DM = {dm} ; score = {score:.4f}")
        scores.append([dm, score])
        
    return scores


def anti_line_noise_median(mat):
    norm = np.median(mat, axis=0)
    mat = mat / norm
    mat = mat - np.min(mat, axis=0)
    return mat