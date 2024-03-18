import cv2
from PyQt6 import QtWidgets, uic
from PyQt6.QtWidgets import QVBoxLayout, QFileDialog
from PyQt6.QtGui import QIcon
import sys
import pyqtgraph as pg
import time
from imageViewPort import ImageViewport
from functools import partial

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
        self.cahnge_lables()
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
        self.import_buttons = [self.ui.importButton, self.ui.importButton_2,]

        # Bind browse_image function to import buttons
        self.bind_buttons(self.import_buttons, self.browse_image)

        # Initialize clear buttons
        self.clear_buttons = [self.ui.clearButton, self.ui.clearButton_2]

        # Bind clear function to clear buttons
        self.bind_buttons(self.clear_buttons, self.clear)

        # Initialize reset buttons
        self.reset_buttons = [self.ui.resetButton, self.ui.resetButton_2,]

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
        if len(buttons) == 5:
            for i, button in enumerate(buttons):
                button.clicked.connect(lambda event, index=i: function(event, index))
        else:
            for i, button in enumerate(buttons):
                button.clicked.connect(lambda index=i: function(index))


    def apply_changes(self):
        current_index = self.ui.hough_comboBox.currentIndex()

        if current_index == 0:
            self.apply_lineHough()
        elif current_index == 1:
            self.apply_circleHough()
        elif current_index == 2:
            self.apply_ellipseHough()


    def handle_hough_combobox(self):
        """
        This function handles the Hough Transform combobox.
        """
        self.cahnge_lables()
        self.handle_hough_sliders()

    def cahnge_lables(self):
        """
        This function changes the labels based on the Hough Transform.
        """
        # time.sleep(0.5)
        current_index = self.ui.hough_comboBox.currentIndex()

        if current_index == 0:
            self.ui.label1.setText("Rho")
            self.ui.label2.setText("Theta")
            self.ui.label3.hide()

        elif current_index == 1:
            self.ui.label1.setText("minRadius")
            self.ui.label2.setText("maxRadius")
            self.ui.label3.setText("minDist")
            self.ui.label3.show()

        else:
            self.ui.label1.setText("minAxis")
            self.ui.label2.setText("maxAxis")
            self.ui.label3.hide()

    def handle_hough_sliders(self):
        combo_idex = self.ui.hough_comboBox.currentIndex()
        print(combo_idex)
        if combo_idex == 0 or combo_idex == 2:
            self.ui.Slider3.hide()
            self.ui.slider3_val.hide()
        else:
            self.ui.Slider3.show()
            self.ui.slider3_val.show()

        self.sliders_limits()

    def sliders_limits(self):

            self.ui.threshSlider.setMinimum(1)  # Set minimum value for Threshold
            self.ui.threshSlider.setMaximum(100)  # Set maximum value for Threshold
            self.ui.threshSlider.setValue(1)  # Set initial value for Threshold

            # For HoughLinesP
            self.ui.label1.setText("Rho")
            self.ui.Slider1.setMinimum(1)  # Set minimum value for Rho
            self.ui.Slider1.setMaximum(10)  # Set maximum value for Rho
            self.ui.Slider1.setValue(1)  # Set initial value for Rho

            self.ui.label2.setText("Theta")
            self.ui.Slider2.setMinimum(1)  # Set minimum value for Theta
            self.ui.Slider2.setMaximum(180)  # Set maximum value for Theta
            self.ui.Slider2.setValue(1)  # Set initial value for Theta

            # For HoughCircles
            self.ui.label1.setText("minRadius")
            self.ui.Slider1.setMinimum(1)  # Set minimum value for minRadius
            self.ui.Slider1.setMaximum(100)  # Set maximum value for minRadius
            self.ui.Slider1.setValue(1)  # Set initial value for minRadius

            self.ui.label2.setText("maxRadius")
            self.ui.Slider2.setMinimum(1)  # Set minimum value for maxRadius
            self.ui.Slider2.setMaximum(100)  # Set maximum value for maxRadius
            self.ui.Slider2.setValue(1)  # Set initial value for maxRadius

            self.ui.label3.setText("minDist")
            self.ui.Slider3.setMinimum(1)  # Set minimum value for minDist
            self.ui.Slider3.setMaximum(100)  # Set maximum value for minDist
            self.ui.Slider3.setValue(1)  # Set initial value for minDist

            # For HoughEllipse
            self.ui.label1.setText("minAxis")
            self.ui.Slider1.setMinimum(1)  # Set minimum value for minAxis
            self.ui.Slider1.setMaximum(100)  # Set maximum value for minAxis
            self.ui.Slider1.setValue(1)  # Set initial value for minAxis

            self.ui.label2.setText("maxAxis")
            self.ui.Slider2.setMinimum(1)  # Set minimum value for maxAxis
            self.ui.Slider2.setMaximum(100)  # Set maximum value for maxAxis
            self.ui.Slider2.setValue(1)  # Set initial value for maxAxis


    def update_label_text(self):
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
            # Check if the index is for the last viewport in the hybrid tab
            if index == 4:
                # Set the image for the last hybrid viewport
                input_port = self.input_ports[index]
                output_port = self.out_ports[index]
                input_port.set_image(self.image_path)
                output_port.set_image(self.image_path, grey_flag=True)
            # Show the image on all viewports except the last hybrid viewport
            else:
                for _, (input_port, output_port) in enumerate(zip(self.input_ports[:-1], self.out_ports[:-1])):
                    input_port.set_image(self.image_path)
                    output_port.set_image(self.image_path, grey_flag=True)



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
        Clear all the input and output ports.

        Args:
            index (int): The index of the port to clear.
        """
        for _, (input_port, output_port) in enumerate(zip(self.input_ports[:-1], self.out_ports[:-1])):
            input_port.clear()  # Clear the input port
            output_port.clear()  # Clear the output port
        self.clear_histographs()  # Clear the histographs


    def reset_image(self, index: int):
        """
        Resets the image at the specified index in the input_ports list.

        Args:
            event: The event triggering the image clearing.
            index (int): The index of the image to be cleared in the input_ports list.
        """
        self.input_ports[index].set_image(self.image_path)
        self.out_ports[index].set_image(self.image_path, grey_flag=True)



    def apply_lineHough(self):
        pass


    def apply_circleHough(self):
        pass


    def apply_ellipseHough(self):
        pass





def main():
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()