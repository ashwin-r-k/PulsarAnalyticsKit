# Pulsar GUI Design Plan
# ----------------------
# Framework: PyQt5 (for flexibility, multiplatform, and multiple pages)

# You can also consider alternatives:
# - Tkinter: Simpler, but less modern
# - DearPyGui: GPU accelerated but newer
# - Streamlit/Gradio: For web-based GUI, less control

# --- Step-by-step implementation plan ---

# 1. Install PyQt5 if not already:
# pip install PyQt5

# 2. Structure:
# MainWindow
# └── QStackedWidget (pages)
#     ├── Page 1: Load Data
#     ├── Page 2: Plot Channel Characteristics
#     ├── Page 3: Compare Channels
#     ├── Page 4: Dynamic Spectrum
#     ├── Page 5: DM Search and Plot
#     ├── Page 6: Folding

# 3. Backend Integration:
# Use your PulsarDataFile class as the backend engine
# GUI will only send parameters and trigger methods

# Sample Scaffold Code:
# GUI Hook for Pulsar Analysis
# ----------------------------
# The GUI below integrates with your pulsar_analysis class. It follows the multi-page design you described.

# verify all the requirements are met in requirements.txt

# if you get errors
# sudo apt install --reinstall libxcb-xinerama0
# sudo apt install libxcb-xinerama0 libxcb1 libxcb-util1 libx11-xcb1 libxrender1 libxi6 libxext6 libxfixes3
# sudo apt install qt5-default qtbase5-dev qt5-qmake



# # Optionally run check at startup
# verify_requirements()

from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
                             QDialog,QHBoxLayout,QTabWidget)


from core.pulsar_analysis import pulsar_analysis
from guibase.generic_plotting_gui import *
from guibase.gui_log import *
from guibase.utils import *


#import pages 
from guibase.LoadDataPage import *
#Page3
from guibase.IntensityMatrixPage import *
from guibase.PulsarPeriod import *
from guibase.DMSearchPage import *
from guibase.AboutPage import *

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

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


    # def run_compare(self):

    #     run_with_feedback(self.compare_btn,load = True)

    #     pulsar = self.main_window.pulsar
    #     percent = self.percent_input.value()
    #     n = int(pulsar.raw_data.shape[0] * percent / 100)

    #     ch0 = pulsar.raw_data[:n, self.channel1_select.value()]
    #     ch1 = pulsar.raw_data[:n, self.channel2_select.value()]
    #     fs = pulsar.sample_rate 

    #     # Instead of showing in matplotlib window, capture the figure
    #     fig = Figure(figsize=(8, 4))

    #     fig = compare_channels(ch0, ch1, fs, label="Comparison of N and S channels")

    #     show_plot(self,fig)
    #     run_with_feedback(self.compare_btn,load = False)

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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pulsar Analysis GUI")

        self.data_file_path = None
        self.pulsar = None

        # self.stack = QStackedWidget()
        # self.setCentralWidget(self.stack)

        # self.page1 = LoadDataPage(self)
        # self.page2 = ChannelAnalysisPage(self)  # Placeholder for next page

        # self.stack.addWidget(self.page1)
        # self.stack.addWidget(self.page2)

        # Create tab widget and add all pages upfront
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # All pages — added immediately
        self.page1 = LoadDataPage(self)
        self.page2 = ChannelAnalysisPage(self)
        self.page3 = IntensityMatrixPage(self)
        self.page4 = PulsarPeriodPage(self)
        self.page5 = DMSearchPage(self)
        # self.page6 = FoldingPage(self)
        # self.page3 = None
        #self.page4 = None
        # self.page5 = None
        self.page6 = None
        self.page7 = AboutPage(self)

        self.tabs.addTab(self.page1, "Load Data")
        self.tabs.addTab(self.page2, "Channel View")
        self.tabs.addTab(self.page3, "Intensity Matrix")
        self.tabs.addTab(self.page4, "Pulsar Period")
        self.tabs.addTab(self.page5, "DM Search")
        self.tabs.addTab(self.page6, "Folding")
        self.tabs.addTab(self.page7, "About")

        self.statusBar().showMessage("Load a file to begin.") # type: ignore
        self.resize(400, 500)
    
        self.log_window = LogDialog(self)  # No 'self'
        # remove always on top flag from the log window
        # self.log_window.setWindowFlags(self.log_window.windowFlags() & ~Qt.WindowStaysOnTopHint)
        self.log_window.setWindowTitle("Application Log")

        # ---------------------------------------------------
        # self.log_window = LogDialog(self)
        self.log_window.show()

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
