from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QApplication
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
import sys
import os
import webbrowser
import platform
import subprocess

from guibase.updater import get_latest_release_info, auto_update_gui

def get_current_version(version_file="version.txt"):
    if os.path.exists(version_file):
        with open(version_file, "r") as f:
            return f.read().strip()
    return "0.0.0"

def open_webbrowser(link: str):
    webbrowser.open(link)

class WelcomePage(QWidget):
    def __init__(self, continue_callback=None):
        super().__init__()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Logo
        logo_label = QLabel()
        logo_path = os.path.join(os.path.dirname(__file__), "logo.png")
        pixmap = QPixmap(logo_path)
        if not pixmap.isNull():
            pixmap = pixmap.scaled(96, 96)
            logo_label.setPixmap(pixmap)
            logo_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
            layout.addWidget(logo_label)

        self.setWindowTitle("Pulsar Analytics Kit - Welcome")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
        self.resize(600, 400)

        # Title
        title = QLabel("Welcome To Pulsar Analytics Kit")
        title.setStyleSheet("font-size: 24px; font-weight: bold;")
        title.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(title)

        # Intro
        intro = QLabel("A scientific data viewer and analysis toolkit for pulsar research.\nDeveloped by Ashwin Kharat.")
        intro.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(intro)

        # Version info and update button
        version_layout = QHBoxLayout()
        current_version = get_current_version()
        latest_version = "?"
        release = get_latest_release_info()
        update_available = False
        if release and "tag_name" in release:
            latest_version = release["tag_name"].lstrip("v")
        current_label = QLabel(f"Current Version: <b>{current_version}</b>")
        latest_label = QLabel(f"Latest Version: <b>{latest_version}</b>")
        version_layout.addWidget(current_label)
        version_layout.addWidget(latest_label)

        # Check if update is available
        if latest_version != "?" and current_version != "0.0.0":
            try:
                def version_tuple(v):
                    return tuple(map(int, (v.split("."))))
                if version_tuple(latest_version) > version_tuple(current_version):
                    update_available = True
            except Exception:
                pass

        # Add update button if update is available
        if update_available:
            update_btn = QPushButton("Update")
            update_btn.setStyleSheet("background-color: orange; font-weight: bold;")
            def do_update():
                updated_file = auto_update_gui(self, current_version=current_version,release=release)
                if updated_file:
                    if platform.system() == "Windows":
                        subprocess.Popen(['start', '', updated_file], shell=True)
                    elif platform.system() == "Linux":
                        os.chmod(updated_file, 0o755)
                        subprocess.Popen([updated_file])
                    sys.exit(0)
            update_btn.clicked.connect(do_update)
            version_layout.addWidget(update_btn)

        layout.addLayout(version_layout)

        # Update status
        update_msg = ""
        if latest_version != "?" and current_version != "0.0.0":
            try:
                def version_tuple(v):
                    return tuple(map(int, (v.split("."))))
                if version_tuple(latest_version) > version_tuple(current_version):
                    update_msg = f"<span style='color:orange;'>Update available!</span>"
                else:
                    update_msg = f"<span style='color:green;'>You are up to date.</span>"
            except Exception:
                update_msg = ""
        update_label = QLabel(update_msg)
        update_label.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        update_label.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(update_label)

        # README and License buttons
        btn_layout = QHBoxLayout()
        readme_btn = QPushButton("Open README")
        readme_btn.clicked.connect(lambda: open_webbrowser("https://github.com/ashwin-r-k/PulsarAnalyticsKit#readme"))
        btn_layout.addWidget(readme_btn)

        license_btn = QPushButton("License")
        license_btn.clicked.connect(lambda: open_webbrowser("https://github.com/ashwin-r-k/PulsarAnalyticsKit/blob/main/LICENSE"))
        btn_layout.addWidget(license_btn)

        sample_btn = QPushButton("Sample Data")
        sample_btn.clicked.connect(lambda: open_webbrowser("https://iitk-my.sharepoint.com/:f:/g/personal/aswinrk24_iitk_ac_in/Eoht_1SoywhDhix_Q4fjXnMBAoxMA_ggCzEvmJknkOgeQg?e=CC5p2E"))
        btn_layout.addWidget(sample_btn)
        layout.addLayout(btn_layout)

        continue_btn = QPushButton("Continue to Application")
        continue_btn.setStyleSheet("background-color: lightblue; font-weight: bold;")
        layout.addWidget(continue_btn)
        if continue_callback:
            continue_btn.clicked.connect(continue_callback)
        else:
            sys.exit(0)

        author = QLabel("© 2025 Ashwin Kharat")
        author.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(author)
        self.setLayout(layout)



# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = WelcomePage()
#     window.show()
#     sys.exit(app.exec_())