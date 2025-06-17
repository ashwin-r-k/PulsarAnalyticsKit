from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
                             QDialog,QHBoxLayout,QTabWidget)
from core.pulsar_analysis import pulsar_analysis
from guibase.generic_plotting_gui import *
from guibase.gui_log import *
from guibase.utils import *


class ChannelAnalysisPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()

        self.percent_input = QSpinBox()
        self.percent_input.setRange(1, 100)
        self.percent_input.setValue(40)

        self.channel_select = QSpinBox()
        self.channel_select.setMinimum(0)

        self.channel1_select = QSpinBox()
        self.channel1_select.setValue(0)
        self.channel2_select = QSpinBox()
        self.channel2_select.setValue(1)

        self.compare_btn = QPushButton("Compare Channels")
        self.compare_btn.clicked.connect(self.run_compare)

        self.plot_btn = QPushButton("Plot Channel Characteristics")
        self.plot_btn.clicked.connect(self.plot_characteristics)

        layout.addWidget(QLabel("Percent of Data to Use"))
        layout.addWidget(self.percent_input)
        # Put channel1_select and channel2_select side by side

        channel_compare_layout = QHBoxLayout()
        channel_compare_layout.addWidget(QLabel("Channel 1"))
        channel_compare_layout.addWidget(self.channel1_select)
        channel_compare_layout.addWidget(QLabel("Channel 2"))
        channel_compare_layout.addWidget(self.channel2_select)
        layout.addLayout(channel_compare_layout)
        layout.addWidget(self.compare_btn)

        layout.addWidget(QLabel("Channel to Plot Characteristics"))
        layout.addWidget(self.channel_select)
        layout.addWidget(self.plot_btn)

        self.setLayout(layout)

    def run_compare(self):
        run_with_feedback(self.compare_btn, load=True)

        pulsar = self.main_window.pulsar
        percent = self.percent_input.value()
        n = int(pulsar.raw_data.shape[0] * percent / 100)

        ch0 = pulsar.raw_data[:n, self.channel1_select.value()]
        ch1 = pulsar.raw_data[:n, self.channel2_select.value()]
        fs = pulsar.sample_rate

        @run_in_thread(callback=self.on_compare_done)
        def compute_compare():
            return compare_channels(ch0, ch1, fs, label="Comparison of N and S channels")

        compute_compare()

    def on_compare_done(self, fig):
        show_plot(self, fig)
        run_with_feedback(self.compare_btn, load=False)
    
    def plot_characteristics(self):
        run_with_feedback(self.plot_btn ,load = True)
        pulsar = self.main_window.pulsar
        channel = self.channel_select.value()
        fig = Plot_characterstics(pulsar, channel)
        show_plot(self,fig)
        run_with_feedback(self.plot_btn ,load = False)

    # def show_plot(self, fig):
    #     dialog = PlotWindow(fig, self)
    #     dialog.exec_()