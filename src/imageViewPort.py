from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QImage, QPainter
import logging
import cv2
from PyQt6.QtWidgets import QWidget


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
