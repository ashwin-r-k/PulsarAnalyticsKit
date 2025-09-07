from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
from matplotlib.figure import Figure
from core.pulsar_analysis import pulsar_analysis
from guibase.utils import *
from guibase.generic_plotting_gui import *
from guibase.gui_log import *

class IntensityMatrixPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.pulsar = None

        layout = QVBoxLayout()
        
        layout.addWidget(QLabel("Dynamic Spectrum"))

        self.compute_button = QPushButton("Compute Intensity Matrix")
        self.compute_button.clicked.connect(self.compute_intensity_matrix)
        layout.addWidget(self.compute_button)


        # Header
        layout.addWidget(QLabel("RFI Mitigation"))

        # RFI Mitigation controls
        rfi_layout = QHBoxLayout()
        self.freq_std_thresh_input = QDoubleSpinBox(); self.freq_std_thresh_input.setDecimals(1); self.freq_std_thresh_input.setValue(3.0)
        self.freq_mean_thresh_input = QDoubleSpinBox(); self.freq_mean_thresh_input.setDecimals(1); self.freq_mean_thresh_input.setValue(1.0)

        self.time_thresh_input = QDoubleSpinBox(); self.time_thresh_input.setDecimals(1); self.time_thresh_input.setValue(5.0)
        self.fill_value_input = QDoubleSpinBox(); self.fill_value_input.setDecimals(1); self.fill_value_input.setValue(0.0)
        rfi_layout.addWidget(QLabel("Freq σ thresh:")); rfi_layout.addWidget(self.freq_std_thresh_input)
        rfi_layout.addWidget(QLabel("Freq μ thresh:")); rfi_layout.addWidget(self.freq_mean_thresh_input)
        rfi_layout.addWidget(QLabel("Time σ thresh:")); rfi_layout.addWidget(self.time_thresh_input)
        rfi_layout.addWidget(QLabel("Fill value:"));   rfi_layout.addWidget(self.fill_value_input)
        layout.addLayout(rfi_layout)

        self.rfi_btn = QPushButton("Apply RFI Mitigation")
        self.rfi_btn.clicked.connect(self.apply_rfi_mitigation)
        layout.addWidget(self.rfi_btn)


        self.label = QLabel("Select Channel to Plot Intensity Matrix")
        layout.addWidget(self.label)

        self.channel_dropdown = QComboBox()
        layout.addWidget(self.channel_dropdown)

        self.plot_button = QPushButton("Plot Intensity Matrix")
        self.plot_button.clicked.connect(self.run_plot_intensity_matrix)
        layout.addWidget(self.plot_button)

        self.setLayout(layout)

    def showEvent(self, a0):
        super().showEvent(a0)
        self.refresh_channel_list()

    def refresh_channel_list(self):
        self.pulsar = self.main_window.pulsar
        self.channel_dropdown.clear()
        if self.pulsar and self.pulsar.channel_names:
            self.channel_dropdown.addItems(self.pulsar.channel_names)

    def compute_intensity_matrix(self):
        run_with_feedback(self.compute_button, load=True)
        self.pulsar = self.main_window.pulsar
        if self.pulsar:
            self.pulsar.compute_intensity_matrix()
        run_with_feedback(self.compute_button, load=False)

    def run_plot_intensity_matrix(self):
        run_with_feedback(self.plot_button, load=True)
        self.pulsar = self.main_window.pulsar
        channel_index = self.channel_dropdown.currentIndex()

        fig = plot_intensity_matrix(self.pulsar,channel=channel_index, dedispersed=False)

        show_plot(self,fig)
        run_with_feedback(self.plot_button, load=False)

    def apply_rfi_mitigation(self):
        run_with_feedback(self.rfi_btn, load=True)
        self.pulsar = self.main_window.pulsar
        if self.pulsar:
            fsth = self.freq_std_thresh_input.value()
            fmth = self.freq_mean_thresh_input.value()
            tth = self.time_thresh_input.value()
            fv  = self.fill_value_input.value()
            # assumes pulsar has RFI_mitigation method
            self.pulsar.RFI_mitigation(freq_ch_std_threshold=fsth,
                                       freq_ch_mean_threshold=fmth,
                                       time_ch_threshold=tth,
                                       fill_value=fv)
        run_with_feedback(self.rfi_btn, load=False)
