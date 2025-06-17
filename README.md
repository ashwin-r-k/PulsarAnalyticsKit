# PulsarAnalytics
Modular pulsar data analysis and visualization pipeline

## To Do Features

- [x] Add mean check condition with the std to remove RFI
- [ ] Test making an interactive plot in gui



## To Do Study Test for Concepts
- [ ] Take Ft for the Wole 1 sec or long data to find the Period of Pulsar.



## To build the standalone package
pip install pyinstaller
pyinstaller --noconfirm --windowed --icon=./guibase/logo.ico main.py

pyinstaller --noconfirm --clean ./mainbuild.spec

mkdir -p AppDir/usr/bin
cp dist/PulsarAnalyticsKit-${{ env.VERSION }}-linux AppDir/usr/bin/PulsarAnalyticsKit
chmod +x AppDir/usr/bin/PulsarAnalyticsKit
mkdir -p AppDir/usr/share/icons/hicolor/256x256/apps
cp ./guibase/logo.png AppDir/usr/share/icons/hicolor/256x256/apps/pulsar.png
mkdir -p AppDir/usr/share/applications

echo "[Desktop Entry]
Version=1.0
X-AppVersion=${{ env.VERSION }}
Name=Pulsar Analytics Kit
Exec=PulsarAnalyticsKit
Icon=pulsar
Type=Application
Categories=Science;Education;" > AppDir/usr/share/applications/pulsar.desktop

ln -s AppDir/usr/share/applications/pulsar.desktop AppDir/pulsar.desktop
ln -s AppDir/usr/share/icons/hicolor/256x256/apps/pulsar.png / AppDir/pulsar.png
chmod 644 AppDir/usr/share/applications/pulsar.desktop

ls -la
echo "Building AppImage LinuxDeploy..."

wget https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-x86_64.AppImage
mv linuxdeploy-x86_64.AppImage linuxdeploy



./linuxdeploy-x86_64.AppImage -v1 \
        --appdir AppDir \
        --output appimage \
        --executable app \
        --desktop-file AppDir/pulsar.desktop \
        --icon-file AppDir/pulsar.png

chmod +x appimagetool

./appimagetool AppDir "PulsarAnalyticsKit-${{ env.VERSION }}.AppImage"

