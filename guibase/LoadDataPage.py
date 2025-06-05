from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
                             QDialog,QHBoxLayout,QTabWidget,QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox,
                            QDoubleSpinBox, QLineEdit, QFileDialog)

from pulsar_analysis import pulsar_analysis
from guibase.utils import run_with_feedback
from guibase.generic_plotting_gui import *
from guibase.gui_log import *


class LoadDataPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()

        self.file_label = QLabel("No file selected")
        self.select_file_btn = QPushButton("Select Data File")
        self.select_file_btn.clicked.connect(self.select_file)

        self.channel_input = QSpinBox()
        self.channel_input.setMinimum(1)
        self.channel_input.setValue(2)
        self.channel_input.valueChanged.connect(self.update_channel_name_inputs)

        # self.center_freq_input = QDoubleSpinBox()
        # self.center_freq_input.setValue(326.5)

        self.center_freq_input = QDoubleSpinBox()
        self.center_freq_input.setRange(0.0000, 10000.0000)  # Allows up to 10 GHz
        self.center_freq_input.setDecimals(4)         # Show 3 decimal places
        self.center_freq_input.setSingleStep(1.0)      # Step size for arrows
        self.center_freq_input.setValue(326.5)
        # self.center_freq_input.setSuffix(" MHz")


        self.bandwidth_input = QDoubleSpinBox()
        self.bandwidth_input.setValue(16.5)

        self.sample_rate_input = QLineEdit("33")

        self.load_btn = QPushButton("Load Data")
        self.load_btn.clicked.connect(self.load_data)

        # Dynamic channel name inputs container
        self.channel_name_layout = QHBoxLayout()
        self.channel_name_inputs = []  # List of QLineEdit objects

        # Main layout
        layout.addWidget(QLabel("Load Pulsar Data"))
        layout.addWidget(self.file_label)
        layout.addWidget(self.select_file_btn)
        layout.addWidget(QLabel("Number of Channels"))
        layout.addWidget(self.channel_input)

        layout.addWidget(QLabel("Channel Names"))
        layout.addLayout(self.channel_name_layout)  # Dynamic name input area

        layout.addWidget(QLabel("Center Frequency (MHz)"))
        layout.addWidget(self.center_freq_input)
        layout.addWidget(QLabel("Bandwidth (MHz)"))
        layout.addWidget(self.bandwidth_input)
        layout.addWidget(QLabel("Sample Rate (10^6 GHz)"))
        layout.addWidget(self.sample_rate_input)

        layout.addWidget(self.load_btn)
        self.setLayout(layout)

        # Initial setup
        self.update_channel_name_inputs()

    def update_channel_name_inputs(self):
        # Clear old inputs
        for input_widget in self.channel_name_inputs:
            input_widget.deleteLater()
        self.channel_name_inputs.clear()

        n = self.channel_input.value()
        for i in range(n):
            line_edit = QLineEdit(f"ch{i}")
            line_edit.setFixedWidth(60)
            self.channel_name_layout.addWidget(line_edit)
            self.channel_name_inputs.append(line_edit)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File", "", "All Files (*)")
        if file_path:
            self.main_window.data_file_path = file_path
            self.file_label.setText(f"Selected File: {file_path}")

    def load_data(self):
        path = self.main_window.data_file_path
        if not path:
            self.file_label.setText("No file selected")
            return

        chans = self.channel_input.value()
        channel_names = [field.text() for field in self.channel_name_inputs]
        
        run_with_feedback(self.load_btn, load=True)

        self.main_window.pulsar = pulsar_analysis(
            file_path=path,
            data_type='ascii',
            channel_names=channel_names,
            n_channels=chans,
            center_freq_MHZ=self.center_freq_input.value(),
            bandwidth_MHZ=self.bandwidth_input.value(),
            sample_rate=float(self.sample_rate_input.text()) * 1e6 # Convert GHz to Hz
        )

        run_with_feedback(self.load_btn, load=False)

        self.main_window.statusBar().showMessage("Data loaded successfully")



class LoadDataPageOld(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()

        self.file_label = QLabel("No file selected")
        self.select_file_btn = QPushButton("Select Data File")
        self.select_file_btn.clicked.connect(self.select_file)

        self.channel_input = QSpinBox()
        self.channel_input.setMinimum(1)
        self.channel_input.setValue(2)

        self.center_freq_input = QDoubleSpinBox()
        self.center_freq_input.setValue(326.5)

        self.bandwidth_input = QDoubleSpinBox()
        self.bandwidth_input.setValue(16.5)

        self.sample_rate_input = QLineEdit("33")

        self.load_btn = QPushButton("Load Data and Go to Next Page")
        self.load_btn.clicked.connect(self.load_data)

        layout.addWidget(QLabel("Load Pulsar Data"))
        layout.addWidget(self.file_label)
        layout.addWidget(self.select_file_btn)
        layout.addWidget(QLabel("Number of Channels"))
        layout.addWidget(self.channel_input)
        layout.addWidget(QLabel("Center Frequency (MHz)"))
        layout.addWidget(self.center_freq_input)
        layout.addWidget(QLabel("Bandwidth (MHz)"))
        layout.addWidget(self.bandwidth_input)
        layout.addWidget(QLabel("Sample Rate (GHz)"))
        layout.addWidget(self.sample_rate_input)
        layout.addWidget(self.load_btn)
        self.setLayout(layout)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Data File")
        if file_path:
            self.file_label.setText(file_path)
            self.main_window.data_file_path = file_path

    def load_data(self):
        path = self.main_window.data_file_path
        chans = self.channel_input.value()
        center = self.center_freq_input.value()
        bw = self.bandwidth_input.value()
        sample_rate = float(self.sample_rate_input.text())

        if not path:
            QMessageBox.critical(self, "Error", "No file selected.")
            return

        run_with_feedback(self.load_btn,load = True)

        self.main_window.pulsar = pulsar_analysis(
            file_path=path,
            data_type='ascii',  # or detect automatically
            channel_names=[f"ch{i}" for i in range(chans)],
            n_channels=chans,
            center_freq_MHZ=center,
            bandwidth_MHZ=bw,
            sample_rate=sample_rate,
        )

        run_with_feedback(self.load_btn,load = False)

        # self.main_window.stack.setCurrentIndex(1)  # Go to next page
