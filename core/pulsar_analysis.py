import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


from scipy.signal import welch, correlate, find_peaks
from scipy import stats
from scipy.stats import linregress
from scipy.optimize import curve_fit

from core.functions import *
# from generic_plotting import *



# I will have a set of functions which can work independently thne the pulsar class will be used to  
# call these functions and do the analysis on the pulsar data file.


class pulsar_analysis:
    # No import needed; use static methods directly from Fn_1ch, e.g., Fn_1ch.compute_channel_intensity_matrix

    def __init__(self, file_path, data_type , channel_names,center_freq_MHZ = 326.5,bandwidth_MHZ = 16.5,skip_rows=1 ,n_channels=2,block_size=512, avg_blocks=60, sample_rate=33e6):
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
        self.skip_rows:int = skip_rows

        # Stokes parameters (I, Q, U, V)
        self.stokes_parameters = np.empty(4, dtype=object)
        
        # Derived polarization parameters
        self.L = None  # Linear polarization
        self.P_frac = None  # Polarization fraction
        self.PA = None  # Polarization angle

        self.load_data()  # Automatically load data upon object creation

        for k, v in vars(self).items():
            if k in ['raw_data', 'intensity_matrix_ch_s', 'dedispersed_ch_s','stokes_parameters']:
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
            #also skip first few lines
            self.raw_data = np.loadtxt(self.file_path, skiprows=self.skip_rows)
            self.n_channels = self.raw_data.shape[1]

        elif self.data_type == 'binary_int32':
            self.raw_data = np.fromfile(self.file_path, dtype=np.int32).reshape(-1, self.n_channels)

        elif self.data_type == 'swan':
            dt = np.dtype([
                ('header', 'S8'), ('Source', 'S10'),
                ('Attenuator_1', '>u1'), ('Attenuator_2', '>u1'),
                ('Attenuator_3', '>u1'), ('Attenuator_4', '>u1'),
                ('LO', '>u2'), ('FPGA', '>u2'),
                ('GPS', '>u2'), ('Packet', '>u4'),
                ('data', '>i1', 1024)
            ])
            self.memmap_file = np.memmap(self.file_path, dtype=dt, mode='r')
            self.data_blocks = self.memmap_file['data']

            reshaped = self.data_blocks.reshape(-1)
            ch0 = (reshaped[0::2].astype(np.int32))
            ch1 = (reshaped[1::2].astype(np.int32))
            self.raw_data = np.stack([ch0, ch1], axis=1)
            self.n_channels = 2
        else:
            raise ValueError("Unsupported data type. Use 'ascii', 'binary_int32' or 'swan'.")        
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


    def RFI_mitigation(self, freq_ch_std_threshold=1.0,freq_ch_mean_threshold=1.0,time_ch_threshold=5.0,fill_value=0):
        """
        Simple RFI mitigation by clipping values above a threshold.
        """
        print("Applying RFI mitigation...")
        for ch in range(self.n_channels):
            self.intensity_matrix_ch_s[ch] = remove_rfi_by_std(self.intensity_matrix_ch_s[ch],
                               chan_sigma_thresh=freq_ch_std_threshold,
                               sample_sigma_thresh=time_ch_threshold,
                               chan_mean_thresh = freq_ch_mean_threshold,
                               fill_value=fill_value)
    

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
            
    def calculate_stokes_parameters(self):
        """
        Calculate Stokes parameters from the intensity matrices of two orthogonal polarization channels.
        This requires that the data has been loaded and intensity matrices have been computed for at least two channels.
        
        The Stokes parameters will be stored in self.stokes_parameters as follows:
        - stokes_parameters[0]: I (total intensity)
        - stokes_parameters[1]: Q (difference between horizontal and vertical polarization)
        - stokes_parameters[2]: U (linear polarization along diagonals)
        - stokes_parameters[3]: V (circular polarization)
        """
        if self.n_channels < 2:
            raise ValueError("At least two orthogonal polarization channels are required to calculate Stokes parameters")
        
        if self.intensity_matrix_ch_s[0] is None or self.intensity_matrix_ch_s[1] is None:
            raise ValueError("Intensity matrices have not been computed. Please call compute_intensity_matrix() first")
        
        # Get the intensity matrices for the first two channels
        ch0_intensity = self.intensity_matrix_ch_s[0]
        ch1_intensity = self.intensity_matrix_ch_s[1]
        
        # Calculate Stokes parameters
        I, Q, U, V = calculate_stokes_parameters(ch0_intensity, ch1_intensity)
        
        # Store the results
        self.stokes_parameters[0] = I  # Total intensity
        self.stokes_parameters[1] = Q  # Linear polarization (horizontal vs vertical)
        self.stokes_parameters[2] = U  # Linear polarization (diagonal)
        self.stokes_parameters[3] = V  # Circular polarization
        
        # Calculate and store derived polarization parameters
        self.L, self.P_frac, self.PA = calculate_polarization_parameters(I, Q, U, V)
        
        print("Stokes parameters calculated successfully")
    
    def plot_stokes_parameters(self, figsize=(14, 10)):
        """
        Plot Stokes parameters and polarization parameters.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        """
        if self.stokes_parameters[0] is None:
            raise ValueError("Stokes parameters have not been calculated. Please call calculate_stokes_parameters() first")
        
        I = self.stokes_parameters[0]  # Total intensity
        Q = self.stokes_parameters[1]  # Linear polarization (horizontal vs vertical)
        U = self.stokes_parameters[2]  # Linear polarization (diagonal)
        V = self.stokes_parameters[3]  # Circular polarization
        
        # Create subplots
        fig, axes = plt.subplots(3, 2, figsize=figsize)
        fig.suptitle('Stokes Parameters', fontsize=16)
        
        # Plot Stokes I
        im0 = axes[0, 0].imshow(I, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='viridis')
        axes[0, 0].set_title('Stokes I (Total Intensity)')
        axes[0, 0].set_xlabel('Frequency Channel')
        axes[0, 0].set_ylabel('Time')
        plt.colorbar(im0, ax=axes[0, 0])
        
        # Plot Stokes Q
        im1 = axes[0, 1].imshow(Q, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='RdBu_r')
        axes[0, 1].set_title('Stokes Q (Linear Polarization H-V)')
        axes[0, 1].set_xlabel('Frequency Channel')
        axes[0, 1].set_ylabel('Time')
        plt.colorbar(im1, ax=axes[0, 1])
        
        # Plot Stokes U
        im2 = axes[1, 0].imshow(U, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='RdBu_r')
        axes[1, 0].set_title('Stokes U (Linear Polarization Diagonal)')
        axes[1, 0].set_xlabel('Frequency Channel')
        axes[1, 0].set_ylabel('Time')
        plt.colorbar(im2, ax=axes[1, 0])
        
        # Plot Stokes V
        im3 = axes[1, 1].imshow(V, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='RdBu_r')
        axes[1, 1].set_title('Stokes V (Circular Polarization)')
        axes[1, 1].set_xlabel('Frequency Channel')
        axes[1, 1].set_ylabel('Time')
        plt.colorbar(im3, ax=axes[1, 1])
        
        # Plot Linear Polarization
        im4 = axes[2, 0].imshow(self.L, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='viridis')
        axes[2, 0].set_title('Linear Polarization |L| = √(Q² + U²)')
        axes[2, 0].set_xlabel('Frequency Channel')
        axes[2, 0].set_ylabel('Time')
        plt.colorbar(im4, ax=axes[2, 0])
        
        # Plot Polarization Angle
        im5 = axes[2, 1].imshow(self.PA, aspect='auto', origin='lower', 
                               interpolation='nearest', cmap='twilight_shifted', 
                               vmin=-90, vmax=90)
        axes[2, 1].set_title('Polarization Angle (degrees)')
        axes[2, 1].set_xlabel('Frequency Channel')
        axes[2, 1].set_ylabel('Time')
        plt.colorbar(im5, ax=axes[2, 1])
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()
        
    def plot_integrated_stokes(self, figsize=(12, 10)):
        """
        Plot time and frequency integrated Stokes parameters.
        
        Parameters
        ----------
        figsize : tuple
            Figure size (width, height) in inches
        """
        if self.stokes_parameters[0] is None:
            raise ValueError("Stokes parameters have not been calculated. Please call calculate_stokes_parameters() first")
        
        I = self.stokes_parameters[0]
        Q = self.stokes_parameters[1]
        U = self.stokes_parameters[2]
        V = self.stokes_parameters[3]
        
        # Time integration (sum over rows)
        I_time = np.sum(I, axis=0)
        Q_time = np.sum(Q, axis=0)
        U_time = np.sum(U, axis=0)
        V_time = np.sum(V, axis=0)
        
        # Frequency integration (sum over columns)
        I_freq = np.sum(I, axis=1)
        Q_freq = np.sum(Q, axis=1)
        U_freq = np.sum(U, axis=1)
        V_freq = np.sum(V, axis=1)
        
        # Create figure with two rows of subplots
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        fig.suptitle('Integrated Stokes Parameters', fontsize=16)
        
        # Time-integrated plots (showing variation with frequency)
        axes[0].plot(I_time, label='I', color='black', linewidth=2)
        axes[0].plot(Q_time, label='Q', color='red', linewidth=1)
        axes[0].plot(U_time, label='U', color='green', linewidth=1)
        axes[0].plot(V_time, label='V', color='blue', linewidth=1)
        axes[0].set_title('Time-Integrated Stokes Parameters')
        axes[0].set_xlabel('Frequency Channel')
        axes[0].set_ylabel('Integrated Intensity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Frequency-integrated plots (showing variation with time)
        axes[1].plot(I_freq, label='I', color='black', linewidth=2)
        axes[1].plot(Q_freq, label='Q', color='red', linewidth=1)
        axes[1].plot(U_freq, label='U', color='green', linewidth=1)
        axes[1].plot(V_freq, label='V', color='blue', linewidth=1)
        axes[1].set_title('Frequency-Integrated Stokes Parameters')
        axes[1].set_xlabel('Time Bin')
        axes[1].set_ylabel('Integrated Intensity')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.show()

