from PyQt5.QtWidgets import QMessageBox, QProgressDialog, QApplication
from PyQt5.QtCore import Qt
import os, platform, requests

REPO = "ashwin-r-k/PulsarAnalyticsKit"
VERSION_FILE = "version.txt"

def get_local_version():
    try:
        with open(VERSION_FILE) as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.0.0"

def get_latest_release_info():
    url = f"https://api.github.com/repos/{REPO}/releases/latest"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        pass
    return None

def find_asset(release_data):
    os_type = platform.system().lower()
    for asset in release_data.get("assets", []):
        name = asset["name"].lower()
        if os_type == "windows" and name.endswith(".exe"):
            return asset
        elif os_type == "linux" and (".appimage" in name or name.endswith(".AppImage")):
            return asset
    return None

def download_asset(asset, version, parent=None):
    url = asset["browser_download_url"]
    name = asset["name"]
    filename = f"{name.rsplit('.', 1)[0]}-{version}.{name.rsplit('.', 1)[-1]}"
    dest_path = os.path.join(os.path.dirname(__file__), filename)

    progress = QProgressDialog("Downloading update...", "Cancel", 0, 100, parent)
    # progress.setWindowModality(Qt.WindowModal)
    progress.setWindowTitle("Updater")
    progress.show()

    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 1))
    downloaded = 0

    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if progress.wasCanceled():
                r.close()
                os.remove(dest_path)
                return None
            f.write(chunk)
            downloaded += len(chunk)
            progress.setValue(int(downloaded * 100 / total))
    progress.setValue(100)
    return filename

def auto_update_gui(parent=None):
    local_version = get_local_version()
    release = get_latest_release_info()

    if not release:
        QMessageBox.warning(parent, "Update Check", "Could not fetch latest release info.")
        return False

    remote_version = release["tag_name"].lstrip("v")

    if local_version != remote_version:
        reply = QMessageBox.question(parent, "Update Available",
                                     f"A new version ({remote_version}) is available.\n"
                                     f"Current: {local_version}\nDo you want to update?",
                                     QMessageBox.Yes | QMessageBox.No)

        if reply == QMessageBox.Yes:
            asset = find_asset(release)
            if asset:
                filename = download_asset(asset, remote_version, parent)
                if filename:
                    QMessageBox.information(parent, "Update Complete",
                                            f"Update downloaded as:\n{filename}")
                    return True
                else:
                    QMessageBox.warning(parent, "Download Cancelled", "Update cancelled by user.")
            else:
                QMessageBox.warning(parent, "No Compatible File", "No suitable file found for this OS.")
    else:
        QMessageBox.information(parent, "Up to Date", "You are already using the latest version.")
    return False
