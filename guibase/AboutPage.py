from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QHBoxLayout, QSpacerItem, QSizePolicy
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class AboutPage(QWidget):
    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout()

        # Title
        title_label = QLabel("About Pulsar Analytics")
        # title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 20px; font-weight: bold;")
        layout.addWidget(title_label)

        # Image and Info Row
        info_layout = QHBoxLayout()

        # Optional Image (replace with your image path)
        image_label = QLabel()
        try:
            pixmap = QPixmap("./guibase/profile-img.png")  # Use your image path here
            # pixmap = pixmap.scaled(120, 120, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            image_label.setPixmap(pixmap)
        except:
            image_label.setText("[No Image]")
            # image_label.setAlignment(Qt.AlignCenter)

        info_layout.addWidget(image_label)

        # Info Text
        text_layout = QVBoxLayout()
        text_layout.addWidget(QLabel("Developer: Ashwin Kharat"))
        text_layout.addWidget(QLabel("Email: aswinrk24@iitk.ac.in"))
        text_layout.addWidget(QLabel("Affiliation: SPASE, IIT Kanpur"))
        text_layout.addWidget(QLabel("GitHub: github.com/ashwin-r-k"))

        info_layout.addLayout(text_layout)
        info_layout.addItem(QSpacerItem(20, 0, QSizePolicy.Expanding, QSizePolicy.Minimum))

        layout.addLayout(info_layout)

        # Thank you message
        thank_you = QLabel("\nThank you for using Pulsar Analytics!\nThis tool was designed to support advanced pulsar data processing in a modular and accessible format.")
        # thank_you.setAlignment(Qt.AlignCenter)
        layout.addWidget(thank_you)

        self.setLayout(layout)
