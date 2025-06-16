from PyQt5.QtWidgets import QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt
import os, platform, requests, sys

REPO = "ashwin-r-k/PulsarAnalyticsKit"
token = os.environ.get("TOKEN")
headers = {"Authorization": f"token {token}"} if token else {}

def get_latest_release_info():
    url = f"https://api.github.com/repos/{REPO}/releases/latest"
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print("Error fetching release info:", e)
    return None

def find_asset(release_data):
    os_type = platform.system().lower()
    for asset in release_data.get("assets", []):
        name = asset["name"].lower()
        print(name)
        if os_type == "windows" and name.endswith(".exe"):
            return asset
        elif os_type == "linux" and (".appimage" in name or name.endswith(".appimage")):
            return asset
    return None

def download_asset(asset, parent=None):
    url = asset["browser_download_url"]
    name = asset["name"]
    dest_path = os.path.join(os.path.dirname(sys.argv[0]), name)

    progress = QProgressDialog("Downloading update...", "Cancel", 0, 100, parent)
    progress.setWindowTitle("Updater")
    progress.setWindowModality(Qt.WindowModality.WindowModal)
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
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                progress.setValue(int(downloaded * 100 / total))
    progress.setValue(100)
    return dest_path

def auto_update_gui(parent=None, current_version=None):
    release = get_latest_release_info()
    if not release:
        QMessageBox.warning(parent, "Update Check", "Could not fetch latest release info.")
        return None

    remote_version = release["tag_name"].lstrip("v")
    asset = find_asset(release)
    if not asset:
        QMessageBox.warning(parent, "No Compatible File", "No suitable file found for this OS.")
        return None

    # Only download if remote_version > current_version
    if current_version is not None:
        def version_tuple(v):
            return tuple(map(int, (v.split("."))))
        try:
            if version_tuple(remote_version) <= version_tuple(current_version):
                QMessageBox.information(parent, "No Update Needed",
                    f"You already have the latest version ({current_version}).")
                return None
        except Exception as e:
            print("Version comparison failed:", e)
            # If version parsing fails, proceed with update

    reply = QMessageBox.question(parent, "Update Available",
                                 f"A new version ({remote_version}) is available.\n"
                                 f"Do you want to download and update now?",
                                 QMessageBox.Yes | QMessageBox.No)
    if reply == QMessageBox.Yes:
        filename = download_asset(asset, parent)
        if filename:
            QMessageBox.information(parent, "Update Complete",
                                    f"Update downloaded as:\n{filename}\n\n"
                                    "The new version will now start. This instance will close.")
            return filename
        else:
            QMessageBox.warning(parent, "Download Cancelled", "Update cancelled by user.")
    return None