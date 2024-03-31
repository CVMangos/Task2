from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog, QMessageBox
from PyQt6.QtGui import QIcon
import sys
from src.imageViewPort import ImageViewport
from src.snake import SnakeContour
from src.parameters import Parameters
from src.Validator import Validator
import cv2


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
        self.ui.threshSlider.valueChanged.connect(self.update_label_text)
        self.ui.Slider1.valueChanged.connect(self.update_label_text)
        self.ui.Slider2.valueChanged.connect(self.update_label_text)
        self.ui.Slider3.valueChanged.connect(self.update_label_text)
        self.ui.applyButton.clicked.connect(self.apply_changes)
        self.ui.clearButton.clicked.connect(self.clear)
        self.ui.applyContour.clicked.connect(self.apply_activeContour)
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


    def reset_image(self, event, index: int):
        """
        Resets the image at the specified index in the input_ports list.

        Args:
            event: The event triggering the image clearing.
            index (int): The index of the image to be cleared in the input_ports list.
        """
        self.input_ports[index].set_image(self.image_path)
        self.out_ports[index].set_image(self.image_path, grey_flag=True)


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

        elif current_index == 1:
            # For HoughCircles
            self.ui.label1.setText("Min Radius")  # Label for min_radius
            self.ui.label2.setText("Max Radius")  # Label for max_radius
            self.ui.label3.setText("Min Distance")  # Label for min_dist
            self.ui.label3.show()  # Show the label for min_dist

        else:
            # For HoughEllipse
            self.ui.label1.setText("Min Axis")  # Label for min_axis
            self.ui.label2.setText("Max Axis")  # Label for max_axis
            self.ui.label3.hide()  # Hide the label for min_dist


    def handle_hough_sliders(self):
        """
        Handles the visibility of the third Hough slider based on the selected
        item from the Hough Transform combobox.

        If the selected item is HoughLinesP or HoughEllipse, the third slider
        is hidden, otherwise it's shown.
        """
        combo_idex = self.ui.hough_comboBox.currentIndex()
        print(combo_idex)
        if combo_idex == 0 or combo_idex == 2:  # HoughLinesP or HoughEllipse
            self.ui.Slider3.hide()  # Hide the third slider
            self.ui.slider3_val.hide()  # Hide the label for min_dist
        else:  # HoughCircles
            self.ui.Slider3.show()  # Show the third slider
            self.ui.slider3_val.show()  # Show the label for min_dist

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

        self.ui.threshSlider.setMinimum(1)  # Set minimum value for Threshold
        self.ui.threshSlider.setMaximum(100)  # Set maximum value for Threshold
        self.ui.threshSlider.setValue(1)  # Set initial value for Threshold

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
        threshold_value = self.ui.threshSlider.value()
        self.ui.slider0_val.setText(f"{threshold_value}")

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

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()

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


    def apply_lineHough(self):
        pass


    def apply_circleHough(self):
        pass


    def apply_ellipseHough(self):
        pass

    def get_snake_params(self):
        """
        Get the parameters for the snake algorithm.

        Returns:
            - window_size (int): The size of the window.
            - points_num (int): The number of points.
            - iterations (int): The number of iterations.
            - alpha (float): The value of alpha.
            - beta (float): The value of beta.
            - gamma (float): The value of gamma.
            - radius (int): The radius.
        """
        validator = Validator()

        # Get the parameters for the active contour algorithm
        original_img = self.input_ports[1].resized_img.copy()  # Create a copy of the original image
        circle_center = validator.validate_center(self.ui.center_)
        radius = validator.validate_raduis(self.ui.radius_)
        window_size = validator.validate_parameter(self.ui.windowSize_, "window size", int)
        iterations = validator.validate_parameter(self.ui.iter_, "iterations", int)
        points_num = validator.validate_parameter(self.ui.pointsNum_, "number of points", int)
        alpha = validator.validate_parameter(self.ui.alpha_, "alpha", float)
        beta = validator.validate_parameter(self.ui.beta_, "beta", float)
        gamma = validator.validate_parameter(self.ui.gamma_, "gamma", float)

        # Draw the circle on the copied image
        cv2.circle(original_img, circle_center, radius, (0, 255, 0), thickness=2)

        # Update the output port with the modified image
        self.out_ports[1].original_img = original_img
        self.out_ports[1].update_display()

        snake_params = [original_img, circle_center, radius, window_size, iterations, points_num, alpha, beta, gamma]

        return snake_params

    
    def apply_activeContour(self):
        """
        Apply the active contour algorithm to the image in the input port and display the result
        in the output port.
        """
        # Store the resulting image in the output port and display it
        output_port = self.out_ports[1]

        # Get the active contour parameters and apply them to the image
        snake_params = self.get_snake_params()
        snake = SnakeContour(*snake_params)
        final_curve, segmented_image = snake.active_contour()

        # Calculate the bounding box and perimeter of the shape
        bounding_box = (
            min(final_curve, key=lambda p: p[0])[0],  # Minimum x-coordinate
            max(final_curve, key=lambda p: p[0])[0],  # Maximum x-coordinate
            min(final_curve, key=lambda p: p[1])[1],  # Minimum y-coordinate
            max(final_curve, key=lambda p: p[1])[1]   # Maximum y-coordinate
        )

        param = Parameters(final_curve)
        primeter = param.calculate_perimeter()
        # Calculate the area of the shape using the bounding box and the shape points
        area = param.calculate_area_free_shape(bounding_box)

        # Get the chain code of the shape
        chain_code = param.get_chain_code()
        print(f"Chain code: {chain_code}") 

        # Store the resulting image in the output port and display it
        output_port.set_image("snake.png")
        output_port.update_display()

        # Display the area and perimeter in the GUI
        self.ui.areaLabel.setText(str(int(area)))
        self.ui.perimeterLabel.setText(str(int(primeter)))




def main():
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()