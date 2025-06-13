# updater.py
import os, io, zipfile, shutil
import requests

REPO = "ashwin-r-k/PulsarAnalyticsKit"  # Replace with your actual repo
BRANCH = "main"
ZIP_URL = f"https://github.com/{REPO}/archive/refs/heads/{BRANCH}.zip"
VERSION_URL = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/version.txt"

def get_local_version():
    try:
        with open("version.txt") as f:
            return f.read().strip()
    except FileNotFoundError:
        return "0.0.0"

def get_remote_version():
    try:
        r = requests.get(VERSION_URL)
        if r.status_code == 200:
            return r.text.strip()
    except:
        return None

def auto_update():
    local_version = get_local_version()
    remote_version = get_remote_version()

    print(f"Local version: {local_version}, Remote version: {remote_version}")
    if remote_version and local_version != remote_version:
        print("🔁 New version found. Updating...")

        response = requests.get(ZIP_URL)
        z = zipfile.ZipFile(io.BytesIO(response.content))
        temp_folder = "update_temp"
        z.extractall(temp_folder)

        folder_inside = next(name for name in os.listdir(temp_folder) if os.path.isdir(os.path.join(temp_folder, name)))

        for root, _, files in os.walk(os.path.join(temp_folder, folder_inside)):
            rel_path = os.path.relpath(root, os.path.join(temp_folder, folder_inside))
            for file in files:
                src = os.path.join(root, file)
                dst = os.path.join(".", rel_path, file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

        shutil.rmtree(temp_folder)
        with open("version.txt", "w") as f:
            f.write(remote_version)
        print("✅ Update complete. Please restart.")
        return True
    else:
        print("✔️ Already up to date.")
        return False
