from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
from matplotlib.figure import Figure
from core.pulsar_analysis import pulsar_analysis
from guibase.utils import *
from guibase.generic_plotting_gui import *
from guibase.gui_log import *


class DMSearchPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.pulsar = None

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Dispersion Measure (DM) Search"))

        self.channel_input_search = QSpinBox()
        self.channel_input_search.setMinimum(0)
        layout.addWidget(QLabel("Channel for DM Search"))
        layout.addWidget(self.channel_input_search)

        self.num_peaks_input = QSpinBox()
        self.num_peaks_input.setMinimum(1)
        self.num_peaks_input.setValue(10)
        layout.addWidget(QLabel("Number of Peaks for Fitting"))
        layout.addWidget(self.num_peaks_input)

        self.dm_min_input = QDoubleSpinBox()
        self.dm_min_input.setDecimals(2)
        self.dm_min_input.setValue(10.0)
        layout.addWidget(QLabel("Minimum DM"))
        layout.addWidget(self.dm_min_input)

        self.dm_max_input = QDoubleSpinBox()
        self.dm_max_input.setDecimals(2)
        self.dm_max_input.setValue(100.0)
        layout.addWidget(QLabel("Maximum DM"))
        layout.addWidget(self.dm_max_input)

        self.tol_input = QDoubleSpinBox()
        self.tol_input.setDecimals(2)
        self.tol_input.setValue(1.0)
        self.tol_input.setSingleStep(0.5)
        layout.addWidget(QLabel("DM Search Tolerance"))
        layout.addWidget(self.tol_input)

        self.search_btn = QPushButton("Run DM Search")
        self.search_btn.clicked.connect(self.run_dm_search)
        layout.addWidget(self.search_btn)

        layout.addWidget(QLabel("Set Best DM and Dedisperse"))

        self.channel_input = QLineEdit("0")
        layout.addWidget(QLabel("Channel Number (use 'all' for all channels)"))
        layout.addWidget(self.channel_input)

        self.dm_input = QDoubleSpinBox()
        self.dm_input.setDecimals(4)
        self.dm_input.setValue(69.750)
        self.dm_input.setSingleStep(0.1)
        layout.addWidget(QLabel("Set DM Value (pc/cm^3)"))
        layout.addWidget(self.dm_input)

        self.dedisperse_btn = QPushButton("Perform Dedispersion")
        self.dedisperse_btn.clicked.connect(self.run_dedispersion_only)
        layout.addWidget(self.dedisperse_btn)

        self.plot_btn = QPushButton("Plot Dedispersed Intensity Matrix")
        self.plot_btn.clicked.connect(self.plot_dedispersed_matrix)
        layout.addWidget(self.plot_btn)

        self.setLayout(layout)

    def run_dm_search(self):
        run_with_feedback(self.search_btn, load=True)
        self.pulsar = self.main_window.pulsar

        channel = self.channel_input_search.value()
        num_peaks = self.num_peaks_input.value()
        to_plot = False
        dm_min = self.dm_min_input.value()
        dm_max = self.dm_max_input.value()
        tol = self.tol_input.value()

        scores = self.pulsar.Auto_dedisperse(channel, num_peaks, to_plot, dm_min, dm_max, tol)
        fig = plot_dm_curve(np.array(scores)[:, 0], np.array(scores)[:, 1])
        show_plot(self,fig)
        run_with_feedback(self.search_btn, load=False)

    def run_dedispersion_only(self):
        run_with_feedback(self.dedisperse_btn, load=True)
        self.pulsar = self.main_window.pulsar

        dm_value = self.dm_input.value()
        channel_value = self.channel_input.text().strip().lower()

        self.pulsar.dedispersion_measure = dm_value

        if channel_value == "all":
            self.pulsar.Manual_dedisperse(DM=dm_value, channel="all")
        else:
            try:
                channel = int(channel_value)
                self.pulsar.Manual_dedisperse(DM=dm_value, channel=channel)
            except ValueError:
                print("Invalid channel input. Must be an integer or 'all'.")

        run_with_feedback(self.dedisperse_btn, load=False)

    def plot_dedispersed_matrix(self):
        run_with_feedback(self.plot_btn, load=True)
        self.pulsar = self.main_window.pulsar

        try:
            channel = int(self.channel_input.text().strip())
            fig = plot_intensity_matrix(self.pulsar,channel=channel, dedispersed=True)
            show_plot(self,fig)
        except ValueError:
            print("Cannot plot for 'all'. Please enter a single channel number to plot.")

        run_with_feedback(self.plot_btn, load=False)

    def show_plot(self, fig):
        self.plot_window = PlotWindow(fig)
        self.plot_window.show()
