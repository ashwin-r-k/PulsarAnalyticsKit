# main.py

# Start GUI only if not updated
from guibase.gui import *
from guibase.updater import auto_update_gui

import subprocess
import sys
import os
import platform
from PyQt5.QtWidgets import QApplication, QMessageBox

from guibase.WelcomePage import WelcomePage
from guibase.gui import MainWindow


# Set platform environment variable before QApplication
if platform.system() == "Windows":
    os.environ['QT_QPA_PLATFORM'] = 'windows'
elif platform.system() == "Linux":
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
else:
    print("Unsupported platform. Defaulting to offscreen.")
    os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def load_current_version(version_file="version.txt"):
    if not os.path.exists(version_file):
        print(f"Version file '{version_file}' not found. Using default version.")
        return "0.0.0"
    with open(version_file, "r") as f:
        version = f.read().strip()
        print(f"Current version: {version}")
        return version

def verify_requirements(requirements_path="requirements.txt"):
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
        pk_name = "\n".join(missing)
        print("Missing packages:\n", pk_name)
        print(f"You can install them with:\n  pip install -r {requirements_path}")
        QMessageBox.information(None, "Missing packages", f"Missing:\n{pk_name}\n\nYou can install them with:\npip install -r {requirements_path}")
    else:
        print("All requirements satisfied.")

if __name__ == '__main__':
    app = QApplication(sys.argv)

    def show_main():
        welcome_page.close()
        welcome_page.main_window = MainWindow()
        welcome_page.main_window.show()

    welcome_page = WelcomePage(continue_callback=show_main)
    welcome_page.show()
    sys.exit(app.exec_())