from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
from matplotlib.figure import Figure
from pulsar_analysis import pulsar_analysis
from guibase.utils import *
from guibase.generic_plotting_gui import *
from guibase.gui_log import *

class PulsarPeriodPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.pulsar = None

        layout = QVBoxLayout()

        self.label = QLabel("Estimate Pulsar Period via Autocorrelation")
        layout.addWidget(self.label)

        self.channel_input = QSpinBox()
        self.channel_input.setMinimum(0)
        layout.addWidget(QLabel("Channel Number:"))
        layout.addWidget(self.channel_input)

        self.plot_button = QPushButton("Plot Autocorrelation")
        self.plot_button.clicked.connect(self.plot_autocorrelation)
        layout.addWidget(self.plot_button)

        self.period_input = QDoubleSpinBox()
        self.period_input.setDecimals(5)
        self.period_input.setValue(89.0)
        self.period_input.setSingleStep(0.01)
        layout.addWidget(QLabel("Set Pulsar Period (ms):"))
        layout.addWidget(self.period_input)

        self.set_button = QPushButton("Set Period in Pulsar Object")
        self.set_button.clicked.connect(self.set_period)
        layout.addWidget(self.set_button)

        self.setLayout(layout)

    def plot_autocorrelation(self):
        run_with_feedback(self.plot_button, load=True)
        self.pulsar = self.main_window.pulsar
        channel = self.channel_input.value()
        fig = analyze_autocorrelation(self.pulsar, channel, label="Pulsar Channel")
        show_plot(self,fig)
        run_with_feedback(self.plot_button, load=False)

    def set_period(self):
        run_with_feedback(self.set_button, load=True)
        self.pulsar = self.main_window.pulsar
        self.pulsar.pulseperiod_ms = self.period_input.value()
        run_with_feedback(self.set_button, load=False)
