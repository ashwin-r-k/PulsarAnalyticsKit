
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from scipy.signal import welch, correlate, find_peaks
from scipy import stats
from scipy.stats import linregress
from scipy.optimize import curve_fit


def random_sample_hist(self, max_samples=100000):
    data = self.raw_data
    if data is None:
        raise ValueError("No data loaded.")

    labels = self.channel_names
    no_chs = int(self.n_channels)
    fig, axes = plt.subplots(1,no_chs = int(self.n_channels), figsize=(14, 5), sharey=True)

    for i, ax in enumerate(axes): # type: ignore
        idx = np.random.choice(data.shape[0], size=min(max_samples, data.shape[0]), replace=False)
        sample = data[idx, i]
        mean, std = np.mean(sample), np.std(sample)
        print(f"{labels[i]} mean = {mean:.3f}, std = {std:.3f}")
        ax.hist(sample, bins=200, alpha=0.7, label=labels[i], color=('green' if i == 0 else 'orange'))
        ax.set_title(f"{labels[i]} Histogram")
        ax.set_xlabel("Amplitude")
        ax.legend()

    axes[0].set_ylabel("Count") # type: ignore
    plt.suptitle("Random Sample Histogram (Separate Channels)")
    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.show()

def plot_intensity_matrix(self, channel, gamma=2.5, dedispersed=False):
    try:
        if dedispersed:
            matrix = self.dedispersed_ch_s[channel]
            title_suffix = " (Dedispersed)"
        else:
            matrix = self.intensity_matrix_ch_s[channel]
            title_suffix = ""
    except:
        raise ValueError("Requested intensity / Desispersed matrix not computed.")

    num_segments, n_freq = matrix.shape
    time_bin_us = (self.block_size * self.avg_blocks / self.sample_rate) * 1e6
    time_extent_ms = num_segments * time_bin_us / 1000

    # Frequency axis
    bandwidth = self.bandwidth_MHZ #16.5  # MHz
    center_freq = self.center_freq_MHZ # 326.5  # MHz
    freq_array = np.linspace(center_freq + bandwidth / 2, center_freq - bandwidth / 2, n_freq)

    plt.figure(figsize=(10, 6))
    plt.imshow(matrix.T, aspect='auto', origin='upper', cmap='turbo',
                norm=colors.PowerNorm(gamma=gamma),
                extent=(0, time_extent_ms, freq_array[-1], freq_array[0]))
    plt.colorbar(label="Log Power")
    plt.xlabel("Time (ms)")
    plt.ylabel("Frequency (MHz)")
    plt.title(f"Dynamic Spectrum (Channel {channel}){title_suffix}")
    yticks = np.linspace(freq_array[0], freq_array[-1], 8)
    plt.yticks(yticks)
    plt.show()

def Plot_characterstics(self, channel):
    ch = self.raw_data[:10000, channel]
    label = self.channel_names[channel]
    color = 'green' if channel == 1 else 'darkorange'

    n_samples = ch.shape[0]
    sample_rate = self.sample_rate
    time = np.arange(n_samples) / sample_rate * 1e3  # ms

    f, Pxx = welch(ch, fs=sample_rate, nperseg=1024)

    fig, axs = plt.subplots(3, 1, figsize=(12, 15))

    axs[0].plot(time, ch, label=label, color=color, alpha=0.7)
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Voltage (V)')
    axs[0].set_title(f'Time Series of Noise Signals ({label})')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].semilogy(f / 1e6, Pxx, label=f'{label} PSD', color=color)
    axs[1].set_xlabel('Frequency (MHz)')
    axs[1].set_ylabel('PSD (V²/Hz)')
    axs[1].set_title(f'Power Spectral Density of {label}')
    axs[1].legend()
    axs[1].grid(True, which='both')

    axs[2].hist(ch, bins=100, alpha=0.6, label=label, color=color, density=True)
    axs[2].set_xlabel('Voltage (V)')
    axs[2].set_ylabel('Probability Density')
    axs[2].set_title(f'Histogram of Voltage Values ({label})')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()

def analyze_autocorrelation(self, channel=1, label=""):
    matrix = self.intensity_matrix_ch_s[channel]
    if matrix is None:
        raise ValueError("Intensity matrix not computed for this channel.")

    ch_label = self.channel_names[channel]
    sample_rate = self.sample_rate
    t_bin_s = (self.block_size * self.avg_blocks) / sample_rate
    n_time, n_freq = matrix.shape

    time = np.arange(n_time) * t_bin_s

    acorrs = []
    periods = []
    for i in range(n_freq):
        ch_freq = matrix[:, i]
        n = len(ch_freq)
        acorr = correlate(ch_freq - np.mean(ch_freq), ch_freq - np.mean(ch_freq), mode='full') / n
        lags = np.arange(-n + 1, n) * t_bin_s
        acorr_centered = acorr[n-1+1:]
        peak_idx = np.argmax(acorr_centered)
        period_estimate = lags[n - 1 + 1 + peak_idx]
        acorrs.append(acorr)
        periods.append(period_estimate)

    avg_acorr = np.mean(acorrs, axis=0)
    avg_period = np.mean(periods)
    lags = np.arange(-n_time + 1, n_time) * t_bin_s

    central_idx = n_freq // 2
    central_acorr = acorrs[central_idx]
    central_period = periods[central_idx]

    fig, axs = plt.subplots(3, 1, figsize=(12, 16))
    fig.suptitle(f'Channel Autocorrelation Analysis (Intensity Matrix): {label or ch_label}', fontsize=16)

    axs[0].plot(time * 1e3, matrix[:, central_idx], color='royalblue')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Intensity')
    axs[0].set_title('Time Series (Central Frequency)')
    axs[0].grid(True)

    axs[1].plot(lags * 1e3, central_acorr, color='purple')
    axs[1].set_xlabel('Lag (ms)')
    axs[1].set_ylabel('Autocorrelation')
    axs[1].set_title('Autocorrelation (Central Frequency)')
    axs[1].axvline(central_period * 1e3, color='red', linestyle='--', 
            label=f'Central Freq Period: {central_period*1e3:.3f} ms')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(lags * 1e3, avg_acorr, color='darkgreen')
    axs[2].set_xlabel('Lag (ms)')
    axs[2].set_ylabel('Autocorrelation')
    axs[2].set_title('Average Autocorrelation (All Frequencies)')
    axs[2].set_yscale('log')
    axs[2].axvline(avg_period * 1e3, color='red', linestyle='--', 
        label=f'Avg Period: {avg_period*1e3:.3f} ms')

    min_lag_idx = np.where(lags > 0)[0][0]
    peaks, properties = find_peaks(avg_acorr[min_lag_idx:])
    peak_lags = lags[min_lag_idx:][peaks]
    peak_heights = avg_acorr[min_lag_idx:][peaks]

    if len(peak_heights) > 0:
        top_idx = np.argsort(peak_heights)[::-1][:3]
        print("Top 3 peak positions (ms):", peak_lags[top_idx] * 1e3)
        for i, idx in enumerate(top_idx):
            axs[2].plot(peak_lags[idx] * 1e3, peak_heights[idx], 'o', color='crimson')
            axs[2].annotate(f"{peak_lags[idx]*1e3:.3f} ms", (peak_lags[idx]*1e3, peak_heights[idx]),
                    textcoords="offset points", xytext=(0,10), ha='center', color='crimson')
    else:
        print("No peaks found in average autocorrelation.")
    axs[2].legend()
    axs[2].grid(True)
#    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



