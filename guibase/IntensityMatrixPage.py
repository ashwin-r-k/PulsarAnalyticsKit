from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox
from matplotlib.figure import Figure
from pulsar_analysis import pulsar_analysis
from guibase.utils import *
from guibase.generic_plotting_gui import *
from guibase.gui_log import *

class IntensityMatrixPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.pulsar = None

        self.layout = QVBoxLayout()

        self.label = QLabel("Select Channel to Plot Intensity Matrix")
        self.layout.addWidget(self.label)

        self.compute_button = QPushButton("Compute Intensity Matrix")
        self.compute_button.clicked.connect(self.compute_intensity_matrix)
        self.layout.addWidget(self.compute_button)

        self.channel_dropdown = QComboBox()
        self.layout.addWidget(self.channel_dropdown)

        self.plot_button = QPushButton("Plot Intensity Matrix")
        self.plot_button.clicked.connect(self.run_plot_intensity_matrix)
        self.layout.addWidget(self.plot_button)

        self.setLayout(self.layout)

    def showEvent(self, event):
        super().showEvent(event)
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
