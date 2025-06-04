from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication

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