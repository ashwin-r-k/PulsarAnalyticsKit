from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtCore import QObject, pyqtSignal, Qt
from ansi2html import Ansi2HTMLConverter
import sys

import traceback
from PyQt5.QtWidgets import QMessageBox

#Error Handling
def global_exception_handler(exctype, value, tb):
    # Format the exception info
    error_msg = ''.join(traceback.format_exception(exctype, value, tb))
    print(error_msg)  # Optional: also log to console or file

    # Show a critical error dialog with the message
    QMessageBox.critical(None, "Unexpected Error", f"An unhandled error occurred:\n\n{str(value)}")

# Set the global exception handler
sys.excepthook = global_exception_handler

# loging Python print statements to a QTextEdit widget in PyQt5
class StreamEmitter(QObject):
    text_written = pyqtSignal(str)

class OutputStream:
    def __init__(self, emitter, orig_stream=None):
        self.emitter = emitter
        self.orig_stream = orig_stream

    def write(self, text):
        if self.orig_stream:
            self.orig_stream.write(text)
        self.emitter.text_written.emit(text)

    def flush(self):
        if self.orig_stream:
            self.orig_stream.flush()


class LogDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Application Log")
        self.resize(400, 500)
        # set location to right corner of the screen
        self.move(900, 100)  # Adjust the position as needed

        layout = QVBoxLayout(self)
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        self.log_output.setStyleSheet("background-color: #1e1e1e; color: white; font-family: monospace;")
        layout.addWidget(self.log_output)


        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self.log_output.clear)
        layout.addWidget(clear_btn)

        self.setLayout(layout)

        self.ansi_converter = Ansi2HTMLConverter(inline=True, dark_bg=True)

        self.emitter = StreamEmitter()
        self.emitter.text_written.connect(self.append_ansi_text)

        sys.stdout = OutputStream(self.emitter, sys.__stdout__)
        sys.stderr = OutputStream(self.emitter, sys.__stderr__)

    def append_ansi_text(self, text):
        html = self.ansi_converter.convert(text, full=False)
        # self.log_output.moveCursor(11)  # QTextCursor.End
        self.log_output.insertHtml(html.replace('\n', '<br>'))
        self.log_output.moveCursor(self.log_output.textCursor().End)
