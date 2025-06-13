# main.py

# Start GUI only if not updated
from gui import *
from updater import auto_update


import subprocess
import sys
import os

os.environ['QT_QPA_PLATFORM'] = 'xcb'

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
        for pkg in missing:
            print("  ", pkg)
        print("You can install them with:")
        print(f"  pip install -r {requirements_path}")
        
    else:
        print("All requirements satisfied.")

# Must check before loading GUI
updated = auto_update()

# Optionally run check at startup
verify_requirements()

if updated:
    import sys
    from PyQt5.QtWidgets import QMessageBox, QApplication
    app = QApplication(sys.argv)
    QMessageBox.information(None, "Update Complete", "The application has been updated. Please restart it.")
    sys.exit(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
