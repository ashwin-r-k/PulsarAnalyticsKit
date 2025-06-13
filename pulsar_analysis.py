import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from scipy.signal import welch, correlate, find_peaks
from scipy import stats
from scipy.stats import linregress
from scipy.optimize import curve_fit

from functions import *
# from generic_plotting import *



# I will have a set of functions which can work independently thne the pulsar class will be used to  
# call these functions and do the analysis on the pulsar data file.


class pulsar_analysis:
    # No import needed; use static methods directly from Fn_1ch, e.g., Fn_1ch.compute_channel_intensity_matrix

    def __init__(self, file_path, data_type , channel_names,center_freq_MHZ = 326.5,bandwidth_MHZ = 16.5 ,n_channels=2,block_size=512, avg_blocks=60, sample_rate=33e6):
        self.file_path = file_path
        self.data_type:str = data_type

        self.n_channels = n_channels
        self.raw_data = np.empty( self.n_channels , dtype=object)
        #self.intensity_matrix_ch_s = None
        self.channel_names:list = channel_names        
        self.block_size = block_size
        self.avg_blocks = avg_blocks
        self.sample_rate = sample_rate  # in Hz
        self.intensity_matrix_ch_s = np.empty( self.n_channels , dtype=object)
        self.dedispersed_ch_s = np.empty( self.n_channels , dtype=object)
        self.dedispersed_choped_ch_s = np.empty(self.n_channels, dtype=object)
        self.folded_ch_s = np.empty(self.n_channels, dtype=object)
        self.center_freq_MHZ = center_freq_MHZ
        self.bandwidth_MHZ = bandwidth_MHZ
        self.pulseperiod_ms:float  
        self.dedispersion_measure :float

        self.load_data()  # Automatically load data upon object creation
        for k, v in vars(self).items():
            if k in ['raw_data', 'intensity_matrix_ch_s', 'dedispersed_ch_s']:
                print(f"\033[1;34m{k}\033[0m shape :  {v.shape if isinstance(v, np.ndarray) else v}")
            else:
                print(f"\033[1;31m{k}\033[0m: {v}")


        if self.channel_names == None or self.n_channels != len(self.channel_names):
            print("No of given channels names and in data didn't match")
            self.channel_names = [f"ch{i}" for i in range(0, self.n_channels )]
        
        # if self.n_channels != n_channels:
        #     print("No of given channels and in data didnt match")
        #     self.intensity_matrix_ch_s = np.zeros(self.n_channels)
        #self.intensity_matrix_ch_s = None


    def load_data(self):
        if self.data_type == 'ascii':
            self.raw_data = np.loadtxt(self.file_path)
        elif self.data_type == 'binary':
            self.raw_data = np.fromfile(self.file_path, dtype=np.int32).reshape(-1, self.n_channels)
            #self.raw_data = temp_data[~np.isnan(temp_data).any(axis=1)]
        else:
            raise ValueError("Unsupported data type. Use 'ascii' or 'binary'.")

        print(f"Given Data is of ndim : {self.raw_data.ndim} . shape : {self.raw_data.shape[1]}")
        self.n_channels = self.raw_data.shape[1]

    def compute_intensity_matrix(self):
        if self.raw_data is None or self.n_channels is None:
            raise ValueError("Data not loaded. Please call load_data() first.")
        
        Intensity_Matrix = []
        
        for i in range(self.n_channels):
            channel_data = self.raw_data[:, i]
            Current_matrix =  compute_channel_intensity_matrix(channel_data, self.block_size, self.avg_blocks, self.sample_rate)
            Intensity_Matrix.append(Current_matrix)
        self.intensity_matrix_ch_s = Intensity_Matrix


    def RFI_mitigation(self, threshold=3):
        """
        Simple RFI mitigation by clipping values above a threshold.
        """
        for ch in range(self.n_channels):
            self.intensity_matrix_ch_s[ch] = rfi_remove(self.intensity_matrix_ch_s[ch], threshold=threshold)
    

    def Auto_dedisperse(self,channel,num_peaks,to_plot,dm_min, dm_max,tol = 1):
        matrix = self.intensity_matrix_ch_s[channel]
        center_freq_MHZ = self.center_freq_MHZ
        bandwidth_MHZ = self.bandwidth_MHZ
        sample_rate = self.sample_rate
        block_size = self.block_size
        avg_blocks = self.avg_blocks
        num_peaks = num_peaks
        tol = tol
        pulseperiod_ms = self.pulseperiod_ms

        if pulseperiod_ms is None:
            raise ValueError("Pulse period not set. Please set pulse period before dedispersing.")
        if dm_min is None or dm_max is None:
            raise ValueError("DM range not set. Please set dm_min and dm_max before dedispersing.")
        if matrix is None:
            raise ValueError("Intensity matrix for the channel is not computed. Please compute intensity matrix first.")

        scores = find_best_dm_Grid(matrix, center_freq_MHZ,bandwidth_MHZ, sample_rate, block_size, avg_blocks,num_peaks,pulseperiod_ms, to_plot, dm_min, dm_max, tol)
        return scores
        # plot_dm_curve(np.array(scores)[:,0], np.array(scores)[:,1])

    def Manual_dedisperse(self,DM, channel):
        if channel == "all":
            for i in range(self.n_channels):
                matrix = self.intensity_matrix_ch_s[i]
                dedispersed = dedisperse(matrix, DM,block_size=self.block_size, avg_blocks=self.avg_blocks
                        , sample_rate=self.sample_rate , bandwidth_MHZ = self.bandwidth_MHZ ,center_freq_MHZ = self.center_freq_MHZ)
                self.dedispersed_ch_s[i] = dedispersed

        elif isinstance(channel, int):
            matrix = self.intensity_matrix_ch_s[channel]
            dedispersed = dedisperse(matrix, DM,block_size=self.block_size, avg_blocks=self.avg_blocks
                        , sample_rate=self.sample_rate , bandwidth_MHZ = self.bandwidth_MHZ ,center_freq_MHZ = self.center_freq_MHZ)
            self.dedispersed_ch_s[channel] = dedispersed
            self.dedispersion_measure = DM

    def Manual_dedisperse_pop(self,DM, channel):
        if channel == "all":
            for i in range(self.n_channels):
                matrix = self.intensity_matrix_ch_s[i]
                dedispersed = dedisperse_pop(matrix, DM,block_size=self.block_size, avg_blocks=self.avg_blocks
                        , sample_rate=self.sample_rate , bandwidth_MHZ = self.bandwidth_MHZ ,center_freq_MHZ = self.center_freq_MHZ)
                self.dedispersed_choped_ch_s[i] = dedispersed

        elif isinstance(channel, int):
            matrix = self.intensity_matrix_ch_s[channel]
            dedispersed = dedisperse_pop(matrix, DM,block_size=self.block_size, avg_blocks=self.avg_blocks
                        , sample_rate=self.sample_rate , bandwidth_MHZ = self.bandwidth_MHZ ,center_freq_MHZ = self.center_freq_MHZ)
            self.dedispersed_ch_s[channel] = dedispersed
        
