# -*- mode: python ; coding: utf-8 -*-

import sys
import os

# Read version from version.txt
with open("./version.txt") as f:
    APP_VERSION = f.read().strip()


# Platform-specific settings
if sys.platform.startswith("win"):
    icon_file = "./guibase/logo.ico"
    hidden_imports = []
    output_name = f"PulsarAnalyticsKit-{APP_VERSION}-win.exe"
elif sys.platform.startswith("linux"):
    icon_file = "./guibase/logo.png"
    hidden_imports = []
    output_name = f"PulsarAnalyticsKit-{APP_VERSION}-linux.APPIMAGE"



a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[("./guibase/profile.png", "guibase")],
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)


exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    name=output_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file
)
