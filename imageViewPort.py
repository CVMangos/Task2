from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QImage, QPainter, QPen, QColor
import logging
import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt

class ImageViewport(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_img = None
        self.resized_img = None
        self.image_path = None
        self.is_drawing = False
        self.points = []  # List to store drawing points

    def set_image(self, image_path, grey_flag=False):
        """
        Set the image for the object.

        Args:
            image_path (str): The path to the image file.

        Returns:
            None
            :param image_path:
            :param grey_flag:
        """
        try:
            # Open the image file 
            image = cv2.imread(image_path)

            if image is None:
                raise FileNotFoundError(f"Failed to load image: {image_path}")

            self.image_path = image_path
            if not grey_flag:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Set the original_img attribute 
            self.original_img = image

            self.update_display()

        except FileNotFoundError as e:
            logging.error(e)
        except Exception as e:
            logging.error(f"Error displaying image: {e}")

    def update_display(self):
        """
        Update the display if the original image is not None.
        """
        if self.original_img is not None:
            self.repaint()

    def paintEvent(self, event):
        """
        Override the paint event to draw the image on the widget.

        Args:
        - self: the widget
        - event: the paint event
        """
        super().paintEvent(event)

        if self.original_img is not None:
            painter_img = QPainter(self)
            height, width = self.original_img.shape[:2]  # Get height and width

            # Check if the image is grayscale or RGB
            if len(self.original_img.shape) == 2:  # Grayscale image
                image_format = QImage.Format.Format_Grayscale8

            else:  # RGB image
                image_format = QImage.Format.Format_RGB888

            # Resize the image while preserving aspect ratio
            aspect_ratio = width / height
            target_width = min(self.width(), int(self.height() * aspect_ratio))
            target_height = min(self.height(), int(self.width() / aspect_ratio))
            self.resized_img = cv2.resize(self.original_img, (target_width, target_height))

            # Calculate the position to center the image
            x_offset = (self.width() - target_width) // 2
            y_offset = (self.height() - target_height) // 2

            # Convert image to QImage
            image = QImage(self.resized_img.data, self.resized_img.shape[1], self.resized_img.shape[0],
                           self.resized_img.strides[0], image_format)

            # Draw the image on the widget with the calculated offsets
            painter_img.drawImage(x_offset, y_offset, image)

            # Draw freehand shape if points exist
            if self.points:
                painter_img.setPen(QPen(QColor(Qt.GlobalColor.red), 2, Qt.PenStyle.SolidLine))
                closed_points = self.points + [self.points[0]]  # Add first point to close the shape
                painter_img.drawPolyline(closed_points)

            # Destroy the painter after use
            del painter_img  # This ensures proper cleanup

    def clear(self):
        """
        This method sets the `original_img` attribute to None, effectively clearing the currently displayed image.
        It then triggers an update of the display to reflect the change.

        Parameters:
            None

        Returns:
            None
        """
        self.original_img = None
        self.repaint()

    def clear_points(self):
        self.points = []
        self.repaint()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = True
            self.points.append(event.pos())  # Add starting point

    def mouseMoveEvent(self, event):
        if self.is_drawing:
            self.points.append(event.pos())  # Add points while dragging
            self.update()  # Trigger repaint to show the line

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_drawing = False
            self.points.append(event.pos())  # Add final point
            self.update()  # Trigger repaint to show the final shape

    def get_freehand_points(self):
        """
        Returns a list of tuples representing the freehand drawing coordinates.

        Returns:
            list: A list of tuples (x, y) representing the drawing positions.
        """
        # Assuming self.points stores QPoint objects
        if not self.points:
            return []  # Handle case with no points
        return [(point.x(), point.y()) for point in self.points]  # Convert QPoints to tuples