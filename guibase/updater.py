import os
import platform
import requests

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
    r = requests.get(url)
    if r.status_code == 200:
        return r.json()
    else:
        print("❌ Failed to fetch release info")
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

def download_asset(asset, version):
    url = asset["browser_download_url"]
    name = asset["name"]
    filename = f"{name.rsplit('.', 1)[0]}-{version}.{name.rsplit('.', 1)[-1]}"
    dest_path = os.path.join(os.path.dirname(__file__), filename)

    print(f"⬇️  Downloading: {name}")
    r = requests.get(url, stream=True)
    with open(dest_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"✅ Saved as: {filename}")
    return filename

def auto_update():
    local_version = get_local_version()
    release = get_latest_release_info()
    if not release:
        return False

    remote_version = release["tag_name"].lstrip("v")
    print(f"Local: {local_version} | Remote: {remote_version}")

    if local_version != remote_version:
        print("🔁 New version found.")
        asset = find_asset(release)
        if asset:
            new_file = download_asset(asset, remote_version)
            print(f"✅ Update complete. New file: {new_file}")
            return True
        else:
            print("⚠️ No suitable asset found for this OS.")
    else:
        print("✔️ Already up to date.")
    return False

