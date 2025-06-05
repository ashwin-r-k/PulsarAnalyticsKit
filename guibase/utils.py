from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget, QWidget, QVBoxLayout, QLabel,
                             QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QComboBox, QLineEdit,
                             QDialog,QHBoxLayout,QTabWidget,QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSpinBox,
                            QDoubleSpinBox, QLineEdit, QFileDialog)

from guibase.generic_plotting_gui import *
from guibase.gui_log import *

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavigationToolbar

def run_with_feedback(button, load=True, reset_delay_ms=2000):
    """
    Adds visual feedback to a button:
    - `load=True` shows 'Loading...' in red.
    - `load=False` shows 'Done' in green and resets after `reset_delay_ms`.
    """
    original_text = button.text()

    if load:
        # button.setEnabled(False)
        button.setText(f"{original_text}  ⏳ Loading...")
        button.setStyleSheet("background-color: red; color: white;")
        QApplication.processEvents()
    else:
        button.setText(f"{original_text.split('⏳')[0].strip()} ✅ Done")
        button.setStyleSheet("background-color: green; color: white;")

        def reset_button():
            button.setEnabled(True)
            button.setText(original_text.split('⏳')[0].strip())
            button.setStyleSheet("")

        # QTimer.singleShot(reset_delay_ms, reset_button)


# class PlotWindow(QMainWindow):
#     def __init__(self, fig, parent=None):
#         super().__init__(parent)
#         self.setWindowTitle("Plot Viewer")
#         self.setMinimumSize(800, 600)

#         # Central widget and layout
#         central_widget = QWidget(self)
#         layout = QVBoxLayout(central_widget)

#         # Canvas and toolbar
#         self.canvas = FigureCanvas(fig)
#         self.toolbar = NavigationToolbar(self.canvas, self)

#         layout.addWidget(self.toolbar)
#         layout.addWidget(self.canvas)

#         self.setCentralWidget(central_widget)
#         self.canvas.draw()  # Ensure rendering

# def show_plot(self, fig):
#     # self.plot_window = PlotWindow(fig)
#     # self.plot_window.show()
#     self.plot_windows.append(PlotWindow(fig)) 
#     self.plot_window.show()

# # def show_plot(self, fig):
# #     self.plot_window = PlotWindow(fig)
# #     self.plot_window.show()


class PlotWindow(QMainWindow):
    def __init__(self, fig: Figure, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot Viewer")
        self.setMinimumSize(800, 600)

        central_widget = QWidget(self)
        layout = QVBoxLayout(central_widget)

        self.canvas = FigureCanvas(fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        self.setCentralWidget(central_widget)
        self.canvas.draw()

active_plot_windows = []

def show_plot(a,fig):
    """
    Display a matplotlib Figure in a standalone PlotWindow
    and keep it from being garbage-collected.
    """
    from matplotlib.figure import Figure
    if not isinstance(fig, Figure):
        raise TypeError("Expected a matplotlib.figure.Figure object")

    window = PlotWindow(fig)
    active_plot_windows.append(window)
    window.show()