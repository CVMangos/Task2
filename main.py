import cv2
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog
from PyQt6.QtGui import QIcon
import sys
from imageViewPort import ImageViewport
from functools import partial
import numpy as np
import snake.snake as snake_utils
import matplotlib.pyplot as plt
from Hough import hough_ellipse


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.init_ui()

    def init_ui(self):
        """
        Initialize the UI by loading the UI page, setting the window title, loading UI elements, and checking a specific UI element.
        """
        # Load the UI Page
        self.ui = uic.loadUi('Mainwindow.ui', self)
        self.setWindowTitle("Image Processing ToolBox")
        self.setWindowIcon(QIcon("icons/image-layer-svgrepo-com.png"))
        self.ui.hough_comboBox.currentIndexChanged.connect(self.handle_hough_combobox)
        self.ui.smoothingSlider.valueChanged.connect(self.update_label_text)
        self.ui.t_low.valueChanged.connect(self.update_label_text)
        self.ui.t_high.valueChanged.connect(self.update_label_text)
        self.ui.Slider1.valueChanged.connect(self.update_label_text)
        self.ui.Slider2.valueChanged.connect(self.update_label_text)
        self.ui.Slider3.valueChanged.connect(self.update_label_text)
        self.ui.applyButton.clicked.connect(self.apply_changes)
        self.ui.clearButton.clicked.connect(self.clear)
        self.ui.applyContour.clicked.connect(self.apply_activeContour)
        self.ui.resetContour.clicked.connect(self.reset_activeContour)
        self.change_labels()
        self.handle_hough_sliders()
        self.load_ui_elements()

    def load_ui_elements(self):
        """
        Load UI elements and set up event handlers.
        """
        # Initialize input and output port lists
        self.input_ports = []
        self.out_ports = []

        # Define lists of original UI view ports, output ports
        self.ui_view_ports = [self.ui.input1, self.ui.input2]

        self.ui_out_ports = [self.ui.output1, self.ui.output2]

        # Create image viewports for input ports and bind browse_image function to the event
        self.input_ports.extend([
            self.create_image_viewport(self.ui_view_ports[i], lambda event, index=i: self.browse_image(event, index))
            for i in range(2)])

        # Create image viewports for output ports
        self.out_ports.extend(
            [self.create_image_viewport(self.ui_out_ports[i], mouse_double_click_event_handler=None) for i in range(2)])

        # Initialize import buttons
        self.import_buttons = [self.ui.importButton, self.ui.importButton_2]

        # Bind browse_image function to import buttons
        self.bind_buttons(self.import_buttons, self.browse_image)

        # Initialize reset buttons
        self.reset_buttons = [self.ui.resetButton]

        # Bind reset_image function to reset buttons
        self.bind_buttons(self.reset_buttons, self.reset_image)

    def bind_buttons(self, buttons, function):
        """
        Bind a function to a list of buttons.

        Args:
            buttons (list): List of buttons to bind the function to.
            function (callable): The function to bind to the buttons.

        Returns:
            None
        """
        for i, button in enumerate(buttons):
            button.clicked.connect(lambda event, index=i: function(event, index))

    def apply_changes(self):
        """
        This function applies the changes to the images based on the current index of the Hough Transform combobox.
        """
        current_index = self.ui.hough_comboBox.currentIndex()

        if current_index == 0:
            # HoughLinesP
            self.apply_lineHough()

        elif current_index == 1:
            # HoughCircles
            self.apply_circleHough()

        elif current_index == 2:
            # HoughEllipse
            self.apply_ellipseHough()

    def handle_hough_combobox(self):
        """
        Handle the Hough Transform combobox change.

        This function is called when the user selects a different item from the Hough
        Transform combobox. It updates the labels and sliders based on the new selection.
        """
        self.change_labels()
        self.handle_hough_sliders()

    def change_labels(self):
        """
        Changes the labels based on the Hough Transform.

        This function updates the labels on the user interface when the user
        selects a different item from the Hough Transform combobox.
        """
        current_index = self.ui.hough_comboBox.currentIndex()

        if current_index == 0:
            # For HoughLinesP
            self.ui.label1.setText("Rho")  # Label for rho
            self.ui.label2.setText("Theta")  # Label for theta
            self.ui.label3.hide()  # Hide the label for min_dist
            self.ui.label4.hide()
            self.ui.label5.hide()

        elif current_index == 1:
            # For HoughCircles
            self.ui.label1.setText("Min Radius")  # Label for min_radius
            self.ui.label2.setText("Max Radius")  # Label for max_radius
            self.ui.label3.setText("Min Dist")  # Label for min_dist
            self.ui.label3.show()  # Show the label for min_dist
            self.ui.label4.hide()
            self.ui.label5.hide()

        else:
            # For HoughEllipse
            self.ui.label1.setText("Label")
            self.ui.label2.setText("Label")
            self.ui.label3.setText("Label")
            self.ui.label4.setText("Label")
            self.ui.label5.setText("Label")
            self.ui.label3.show()
            self.ui.label4.show()
            self.ui.label5.show()

    def handle_hough_sliders(self):
        """
        Handles the visibility of the third Hough slider based on the selected
        item from the Hough Transform combobox.

        If the selected item is HoughLinesP or HoughEllipse, the third slider
        is hidden, otherwise it's shown.
        """
        combo_idex = self.ui.hough_comboBox.currentIndex()
        print(combo_idex)
        if combo_idex == 0:  # HoughLinesP or HoughEllipse
            self.ui.Slider3.hide()  # Hide the third slider
            self.ui.slider3_val.hide()  # Hide the label for min_dist
            self.ui.Slider4.hide()
            self.ui.slider4_val.hide()
            self.ui.Slider5.hide()
            self.ui.slider5_val.hide()
        elif combo_idex == 1:
            self.ui.Slider3.show()  # Hide the third slider
            self.ui.slider3_val.show()  # Hide the label for min_dist
            self.ui.Slider4.hide()
            self.ui.slider4_val.hide()
            self.ui.Slider5.hide()
            self.ui.slider5_val.hide()
        else:  # HoughCircles
            self.ui.Slider3.show()  # Show the third slider
            self.ui.slider3_val.show()  # Show the label for min_dist
            self.ui.Slider4.show()
            self.ui.slider4_val.show()
            self.ui.Slider5.show()
            self.ui.slider5_val.show()

        self.sliders_limits()  # Set the limits for the sliders

    def sliders_limits(self):
        """
        This function sets the limits for the sliders based on the selected item
        from the Hough Transform combobox.

        The limits are set to the following values:
            - Threshold: 1 - 100
            - HoughLinesP: rho: 1 - 10, theta: 1 - 180
            - HoughCircles and HoughEllipses: min_radius: 1 - 100,
                                             max_radius: 1 - 100,
                                             min_dist: 1 - 100
        """
        current_index = self.ui.hough_comboBox.currentIndex()

        self.ui.smoothingSlider.setMinimum(1)  # Set minimum value for Threshold
        self.ui.smoothingSlider.setMaximum(100)  # Set maximum value for Threshold
        self.ui.smoothingSlider.setValue(1)  # Set initial value for Threshold

        self.ui.t_low.setMinimum(1)
        self.ui.t_low.setMaximum(100)
        self.ui.t_low.setValue(1)

        self.ui.t_high.setMinimum(1)
        self.ui.t_high.setMaximum(100)
        self.ui.t_high.setValue(1)

        # For HoughLinesP
        if current_index == 0:
            # "Rho"
            self.ui.Slider1.setMinimum(1)  # Set minimum value for Rho
            self.ui.Slider1.setMaximum(10)  # Set maximum value for Rho
            self.ui.Slider1.setValue(1)  # Set initial value for Rho

            # "Theta"
            self.ui.Slider2.setMinimum(1)  # Set minimum value for Theta
            self.ui.Slider2.setMaximum(180)  # Set maximum value for Theta
            self.ui.Slider2.setValue(1)  # Set initial value for Theta

        # For HoughCircles and HoughEllipses
        if current_index == 1 or current_index == 2:
            self.ui.Slider1.setMinimum(1)  # Set minimum value for min_radius
            self.ui.Slider1.setMaximum(100)  # Set maximum value for min_radius
            self.ui.Slider1.setValue(1)  # Set initial value for min_radius

            self.ui.Slider2.setMinimum(1)  # Set minimum value for max_radius
            self.ui.Slider2.setMaximum(100)  # Set maximum value for max_radius
            self.ui.Slider2.setValue(1)  # Set initial value for max_radius

            self.ui.Slider3.setMinimum(1)  # Set minimum value for min_dist
            self.ui.Slider3.setMaximum(100)  # Set maximum value for min_dist
            self.ui.Slider3.setValue(1)  # Set initial value for min_dist

    def update_label_text(self):
        """
        Updates the label text based on the current value of the sliders.

        This function is connected to the slider valueChanged signal,
        and is called whenever the value of a slider changes.
        It updates the text of the label next to the slider to display
        the current value of the slider.
        """
        current_index = self.ui.hough_comboBox.currentIndex()
        # For Threshold
        smoothing_value = self.ui.smoothingSlider.value()
        self.ui.smoothing_val.setText(f"{smoothing_value}")

        t_low = self.ui.t_low.value()
        self.ui.t_low_val.setText(f"{t_low}")

        t_high = self.ui.t_high.value()
        self.ui.t_high_val.setText(f"{t_high}")

        if current_index == 0:
            # For HoughLinesP
            rho_value = self.ui.Slider1.value()
            theta_value = self.ui.Slider2.value()
            self.ui.slider1_val.setText(f"{rho_value}")
            self.ui.slider2_val.setText(f"{theta_value}")

        elif current_index == 1:
            # For HoughCircles
            min_radius_value = self.ui.Slider1.value()
            max_radius_value = self.ui.Slider2.value()
            min_dist_value = self.ui.Slider3.value()
            self.ui.slider1_val.setText(f"{min_radius_value}")
            self.ui.slider2_val.setText(f"{max_radius_value}")
            self.ui.slider3_val.setText(f"{min_dist_value}")

        elif current_index == 2:
            # For HoughEllipse
            min_axis_value = self.ui.Slider1.value()
            max_axis_value = self.ui.Slider2.value()
            self.ui.slider1_val.setText(f"{min_axis_value}")
            self.ui.slider2_val.setText(f"{max_axis_value}")

    ###################################################################################
    #               Browse Image Function and Viewports controls                      #
    ###################################################################################

    def browse_image(self, event, index: int):
        """
        Browse for an image file and set it for the ImageViewport at the specified index.

        Args:
            event: The event that triggered the image browsing.
            index: The index of the ImageViewport to set the image for.
        """
        # Define the file filter for image selection
        file_filter = "Raw Data (*.png *.jpg *.jpeg *.jfif)"

        # Open a file dialog to select an image file
        self.image_path, _ = QFileDialog.getOpenFileName(self, 'Open Image File', './', filter=file_filter)

        # Check if the image path is valid and the index is within the range of input ports
        if self.image_path and 0 <= index < len(self.input_ports):
            # Set the image for the last hybrid viewport
            input_port = self.input_ports[index]
            output_port = self.out_ports[index]
            input_port.set_image(self.image_path)
            output_port.set_image(self.image_path, grey_flag=True)
            input_port.clear_points()
            output_port.clear_points()

    def create_viewport(self, parent, viewport_class, mouse_double_click_event_handler=None):
        """
        Creates a viewport of the specified class and adds it to the specified parent widget.

        Args:
            parent: The parent widget to which the viewport will be added.
            viewport_class: The class of the viewport to be created.
            mouse_double_click_event_handler: The event handler function to be called when a mouse double-click event occurs (optional).

        Returns:
            The created viewport.

        """
        # Create a new instance of the viewport_class
        new_port = viewport_class(self)

        # Create a QVBoxLayout with parent as the parent widget
        layout = QVBoxLayout(parent)

        # Add the new_port to the layout
        layout.addWidget(new_port)

        # If a mouse_double_click_event_handler is provided, set it as the mouseDoubleClickEvent handler for new_port
        if mouse_double_click_event_handler:
            new_port.mouseDoubleClickEvent = mouse_double_click_event_handler

        # Return the new_port instance
        return new_port

    def create_image_viewport(self, parent, mouse_double_click_event_handler):
        """
        Creates an image viewport within the specified parent with the provided mouse double click event handler.
        """
        return self.create_viewport(parent, ImageViewport, mouse_double_click_event_handler)

    def clear(self, index: int):
        """
        Clear the specifed input and output ports.

        Args:
            index (int): The index of the port to clear.
        """

        self.input_ports[index].clear()
        self.out_ports[index].clear()

    def reset_image(self, index: int):
        """
        Resets the image at the specified index in the input_ports list.

        Args:
            event: The event triggering the image clearing.
            index (int): The index of the image to be cleared in the input_ports list.
        """
        self.input_ports[index].set_image(self.image_path)
        self.out_ports[index].set_image(self.image_path, grey_flag=True)

    def reset_activeContour(self):
        input_port = self.input_ports[1]
        output_port = self.out_ports[1]
        input_port.clear_points()
        output_port.clear_points()

    def apply_lineHough(self):
        pass

    def apply_circleHough(self):
        pass

    def apply_ellipseHough(self):
        input_port = self.input_ports[0]
        output_port = self.out_ports[0]
        img = input_port.resized_img.copy()
        detected_ellipses = hough_ellipse(img)
        output_port.original_img = detected_ellipses
        output_port.update_display()
        print("done")

    def apply_activeContour(self):
        points = self.out_ports[1].get_freehand_points()
        # print(f"Points: {points}")
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        # print(f"Xs: {xs}", f"Ys: {ys}")
        # TODO: valudate null
        alpha = float(self.ui.alpha_.text())
        beta = float(self.ui.beta_.text())
        gamma = float(self.ui.gamma_.text())
        iterations = int(self.ui.iterations.text())
        num_points = int(self.ui.points_number.text())
        window_size = int(self.ui.window_size.text())

        img = self.input_ports[1].resized_img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        circle_center = img.shape[0] // 2, img.shape[1] // 2
        snake_curve, output_img = snake_utils.active_contour_from_circle(img, circle_center, circle_radius=194,
                                                                         alpha=alpha, beta=beta,
                                                                         gamma=gamma, num_iterations=iterations,
                                                                         window_size=window_size
                                                                         , num_points=num_points)
        output_port = self.out_ports[1]
        output_port.original_img = output_img
        output_port.update_display()
        print("done")


def main():
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
