# main.py

# Start GUI only if not updated
from guibase.gui import *
from guibase.updater import auto_update

import subprocess
import sys
import os
import platform
from PyQt5.QtWidgets import QApplication, QMessageBox
# Set up QApplication early for dialogs
app2 = QApplication(sys.argv)

# Ask user if they want to check for updates
reply = QMessageBox.question(
    None,
    "Check for Updates",
    "Do you want to check for updates before starting?",
    QMessageBox.Yes | QMessageBox.No,
    QMessageBox.No
)

updated = False
# If the user chooses to check for updates, run the auto_update function

# if reply == QMessageBox.Yes:
#     updated = auto_update()
# else:
#     updated = False

if platform.system() == "Windows":
    os.environ['QT_QPA_PLATFORM'] = 'windows'  # Often not needed on Windows
    QMessageBox.information(None, "Windows Platform Detected", "Running on Windows platform.")
elif platform.system() == "Linux":
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    QMessageBox.information(None, "Linux Platform Detected", "Running on Linux with xcb platform.")
else:
    print("Unsupported platform. Defaulting to offscreen.")
    QMessageBox.information(None, "Unsupported Platform", "This application is not supported on your platform. Defaulting to offscreen mode.")
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Fallback option



def verify_requirements(requirements_path="requirements.txt"):
    """
    Checks if all packages in requirements.txt are installed.
    Prints missing packages.
    """
    if not os.path.exists(requirements_path):
        print(f"Requirements file '{requirements_path}' not found.")
        return

    with open(requirements_path, "r") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    missing = []
    for req in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "show", req.split("==")[0]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            missing.append(req)

    if missing:
        print("Missing packages:")
        pk_name = ""
        for pkg in missing:
            print("  ", pkg)
            pk_name += pkg + " "
        print("You can install them with:")
        print(f"  pip install -r {requirements_path}")
        QMessageBox.information(None, "Missing packages", " Missingpk_name \n pip install -r {requirements_path}")
    else:
        print("All requirements satisfied.")



# Optionally run check at startup
verify_requirements()

if updated:
    # import sys
    # from PyQt5.QtWidgets import QMessageBox, QApplication
    # app = QApplication(sys.argv)
    QMessageBox.information(None, "Update Complete", "The application has been updated. Please restart it.")
    sys.exit(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
