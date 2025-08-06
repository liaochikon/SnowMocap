import sys
import cv2
import numpy as np
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMessageBox

def calculate_physical_square_size(square_size_pixels, dpi=300):
    mm_per_inch = 25.4
    square_size_mm = (square_size_pixels / dpi) * mm_per_inch
    return square_size_mm

def generate_checkerboard(width, height, square_size):
    a4_width = 2480  # 210mm * 300 DPI / 25.4 mm/inch
    a4_height = 3508  # 297mm * 300 DPI / 25.4 mm/inch

    checkerboard_width = width * square_size
    checkerboard_height = height * square_size

    if checkerboard_width > a4_width or checkerboard_height > a4_height:
        return False, "Checkerboard size exceeds A4 size!"

    offset_x = (a4_width - checkerboard_width) // 2
    offset_y = (a4_height - checkerboard_height) // 2
    image = np.ones((a4_height, a4_width), dtype=np.uint8) * 255

    for i in range(height):
        for j in range(width):
            if (i + j) % 2 == 0:
                image[
                    offset_y + i * square_size : offset_y + (i + 1) * square_size,
                    offset_x + j * square_size : offset_x + (j + 1) * square_size
                ] = 0  

    cv2.imwrite('checkerboard_a4.png', image)
    square_size_mm = calculate_physical_square_size(square_size)

    checkerboard_a4_info = {
        "width" : width,
        "height" : height,
        "square_size_mm" : square_size_mm
    }
    with open("checkerboard_a4.json", "w") as outfile:
        outfile.write(json.dumps(checkerboard_a4_info))

    return True, f"Image saved successfully: checkerboard_a4.png\nPhysical size of each square: {square_size_mm:.2f} mm x {square_size_mm:.2f} mm"

class CheckerboardApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Checkerboard Generator")
        self.setGeometry(100, 100, 400, 200)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        width_layout = QHBoxLayout()
        self.width_label = QLabel("Checkerboard Width (number of squares):")
        self.width_input = QLineEdit("6")
        width_layout.addWidget(self.width_label)
        width_layout.addWidget(self.width_input)
        layout.addLayout(width_layout)

        height_layout = QHBoxLayout()
        self.height_label = QLabel("Checkerboard Height (number of squares):")
        self.height_input = QLineEdit("9")
        height_layout.addWidget(self.height_label)
        height_layout.addWidget(self.height_input)
        layout.addLayout(height_layout)

        square_size_layout = QHBoxLayout()
        self.square_size_label = QLabel("Square Size (pixels):")
        self.square_size_input = QLineEdit("300")
        square_size_layout.addWidget(self.square_size_label)
        square_size_layout.addWidget(self.square_size_input)
        layout.addLayout(square_size_layout)

        self.generate_button = QPushButton("Generate Checkerboard")
        self.generate_button.clicked.connect(self.generate)
        layout.addWidget(self.generate_button)

    def generate(self):
        try:
            width = int(self.width_input.text())
            height = int(self.height_input.text())
            square_size = int(self.square_size_input.text())

            if width <= 0 or height <= 0 or square_size <= 0:
                QMessageBox.warning(self, "Input Error", "Please enter positive integers!")
                return

            success, message = generate_checkerboard(width, height, square_size)
            if success:
                QMessageBox.information(self, "Success", message)
            else:
                QMessageBox.warning(self, "Error", message)

        except ValueError:
            QMessageBox.warning(self, "Input Error", "Please enter valid integer integers!  ")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CheckerboardApp()
    window.show()
    sys.exit(app.exec_())