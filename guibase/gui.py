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
from guibase.ChannelAnalysisPage import *

from guibase.WelcomePage import WelcomePage
from guibase.updater import auto_update_gui

# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.figure import Figure

# from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
# from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pulsar Analysis GUI")
        self.data_file_path = None
        self.pulsar = None

        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.log_window = LogDialog(self)
        self.log_window.setWindowTitle("Application Log")
        self.log_window.show()

        # Add your other pages as before...
        self.page1 = LoadDataPage(self)
        self.page2 = ChannelAnalysisPage(self)
        self.page3 = IntensityMatrixPage(self)
        self.page4 = PulsarPeriodPage(self)
        self.page5 = DMSearchPage(self)
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
    

# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = MainWindow()
#     window.show()
#     sys.exit(app.exec_())
