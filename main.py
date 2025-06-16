# main.py

# Start GUI only if not updated
from guibase.gui import *
from guibase.updater import auto_update_gui

import subprocess
import sys
import os
import platform
from PyQt5.QtWidgets import QApplication, QMessageBox
# Set up QApplication early for dialogs
app2 = QApplication(sys.argv)
if platform.system() == "Windows":
    os.environ['QT_QPA_PLATFORM'] = 'windows'  # Often not needed on Windows
    # QMessageBox.information(None, "Windows Platform Detected", "Running on Windows platform.")
elif platform.system() == "Linux":
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    # QMessageBox.information(None, "Linux Platform Detected", "Running on Linux with xcb platform.")
else:
    print("Unsupported platform. Defaulting to offscreen.")
    # QMessageBox.information(None, "Unsupported Platform", "This application is not supported on your platform. Defaulting to offscreen mode.")
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Fallback option

# Ask user if they want to check for updates
reply = QMessageBox.question(
    None,
    "Check for Updates",
    "Do you want to check for updates before starting?",
    QMessageBox.Yes | QMessageBox.No,
    QMessageBox.No
)

# load the current version from the version.txt
def load_current_version(version_file="version.txt"):
    """
    Load the current version from a file.
    Returns the version as a string.
    """
    if not os.path.exists(version_file):
        return "0.0.0"  # Default version if file doesn't exist
    with open(version_file, "r") as f:
        return f.read().strip()

current_version = load_current_version(version_file="version.txt")

# If the user chooses to check for updates, run the auto_update function
updated_file = None
if reply == QMessageBox.Yes:
    updated_file = auto_update_gui(None, current_version=current_version)  # Replace with your actual version

if updated_file:
    # Launch the new file and exit this one
    import subprocess
    import platform
    if platform.system() == "Windows":
        subprocess.Popen(['start', '', updated_file], shell=True)
    elif platform.system() == "Linux":
        # subprocess.Popen(['chmod', '+x', updated_file])
        os.chmod(updated_file, 0o755)
        subprocess.Popen([updated_file])
    sys.exit(0)



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



if __name__ == '__main__':
    print("Starting GUI application...")
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
