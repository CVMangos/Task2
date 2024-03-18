import sys
import cv2
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QFileDialog
from functools import partial


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.image_label = QLabel()
        self.load_button = QPushButton("Load Image")
        self.load_button.clicked.connect(self.load_image)

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.image_label)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.image = None

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
        if file_dialog.exec():
            file_path = file_dialog.selectedFiles()[0]
            self.image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.process_image()

    def process_image(self):
        # Perform Hough Transform
        lines_image = self.detect_lines(self.image)
        circles_image = self.detect_circles(self.image)
        ellipses_image = self.detect_ellipses(self.image)

        # Convert images to QPixmap for display
        lines_pixmap = self.convert_cvimage_to_qpixmap(lines_image)
        circles_pixmap = self.convert_cvimage_to_qpixmap(circles_image)
        ellipses_pixmap = self.convert_cvimage_to_qpixmap(ellipses_image)

        # Display images
        self.display_image(lines_pixmap, "Detected Lines")
        self.display_image(circles_pixmap, "Detected Circles")
        self.display_image(ellipses_pixmap, "Detected Ellipses")

    def detect_lines(self, image):
        # Apply edge detection
        edges = cv2.Canny(image, 50, 150)

        # Perform Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, 1 * 3.14159 / 180, 100, minLineLength=100, maxLineGap=10)

        # Draw detected lines on a blank image
        lines_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        return lines_image

    def detect_circles(self, image):
        # Apply Hough Circle Transform
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

        # Draw detected circles on a blank image
        circles_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if circles is not None:
            circles = circles[0]
            for circle in circles:
                x, y, r = circle
                cv2.circle(circles_image, (int(x), int(y)), int(r), (0, 255, 0), 2)

        return circles_image

    def detect_ellipses(self, image):
        # Find contours
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw detected ellipses on a blank image
        ellipses_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(ellipses_image, ellipse, (0, 255, 0), 2)

        return ellipses_image

    def convert_cvimage_to_qpixmap(self, image):
        height, width, channel = image.shape
        bytes_per_line = 3 * width
        qimage = QImage(image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        return QPixmap.fromImage(qimage)

    def display_image(self, pixmap, window_title):
        label = QLabel()
        label.setPixmap(pixmap)
        label.setWindowTitle(window_title)
        label.show()


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
