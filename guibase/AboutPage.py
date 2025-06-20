from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5 import QtCore
from PyQt5.QtCore import Qt
import sys
import os
# import images_rc  # This line loads the Qt resources

# def resource_path(relative_path):
#     base_path = getattr(sys, '_MEIPASS', os.path.abspath("."))
#     return os.path.join(base_path, relative_path)


class AboutPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()

        # Title
        title_label = QLabel("About Pulsar Analytics")
        about = QLabel("\nThis tool was designed to support advanced pulsar data processing in a modular and accessible format.")

        # title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)
        layout.addWidget(about)
        github_code = QLabel('Code Link: '+'<a href="https://github.com/ashwin-r-k/PulsarAnalyticsKit">PulsarAnalyticsKit</a>')
        github_code.setOpenExternalLinks(True)
        layout.addWidget(github_code)

        # Image and Info Row
        info_layout = QHBoxLayout()

        # Image loading using Qt resource system
        image_label = QLabel()
        # pixmap = QPixmap(":/images/guibase/profile.png")
        pixmap = QPixmap("./guibase/profile.png")

        if not pixmap.isNull():
            pixmap = pixmap.scaled(120, 120)
            image_label.setPixmap(pixmap)
        else:
            image_label.setText("[No Image]")

        info_layout.addWidget(image_label)

        # Info Text
        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("Developer: Ashwin Raju Kharat"))

        # Add website and other links as clickable
        website = QLabel('Website: '+'<a href="https://ashwinrk.com"> ashwinrk.com </a>')
        website.setOpenExternalLinks(True)
        text_layout.addWidget(website)

        linkedin = QLabel('LinkedIn: '+'<a href="https://www.linkedin.com/in/ashwinlrk/"> ashwinlrk/</a>')
        linkedin.setOpenExternalLinks(True)
        text_layout.addWidget(linkedin)

        email = QLabel('Email: '+'<a href="mailto:ashwin@ashwinrk.com">ashwin@ashwinrk.com</a>')
        email.setOpenExternalLinks(True)
        text_layout.addWidget(email)

        affil = QLabel('Affiliation: '+'<a href="https://iitk.ac.in/space/"> SPASE, IIT Kanpur')
        affil.setOpenExternalLinks(True)
        text_layout.addWidget(affil)


        github = QLabel('GitHub: '+'<a href="https://github.com/ashwin-r-k">ashwin-r-k</a>')
        github.setOpenExternalLinks(True)
        text_layout.addWidget(github)

        info_layout.addLayout(text_layout)
        info_layout.addItem(QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        layout.addLayout(info_layout)

        # Thank you message
        thank_you = QLabel("\nThank you for using Pulsar Analytics!")
        # thank_you.setAlignment(Qt.AlignCenter)
        layout.addWidget(thank_you)

        self.setLayout(layout)