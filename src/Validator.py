from PyQt6.QtWidgets import QWidget, QMessageBox, QInputDialog


class Validator(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def show_error_message(self, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Error")
        msg_box.setText(message)
        msg_box.exec()

    def validate_parameter(self, lineEdit, parameter, param_type=int):
        """
        Validates the parameter entered by the user.

        This function prompts the user to enter a positive integer or float value for the specified parameter
        and repeatedly asks for a new parameter until a valid positive integer or float is entered. 
        If the entered parameter is not a positive integer or float, an error message is displayed and the 
        user is prompted to enter a different parameter. If the user cancels the input dialog, None 
        is returned.

        Args:
            lineEdit (QLineEdit): The QLineEdit widget for the parameter.
            parameter (str): The name of the parameter being validated.
            param_type (type): The type of parameter to validate (int or float). Defaults to int.

        Returns:
            int or float or None: The valid parameter value entered by the user, or None if the user cancels the input dialog.
        """
        parameter = None

        while True:
            try:
                parameter_ = param_type(lineEdit.text())
                if parameter_ < 0:
                    raise ValueError
            except ValueError:
                self.show_error_message(f"{parameter} must be a positive {'integer' if param_type == int else 'float'}.")

                # Prompt user to enter a different parameter
                if param_type == int:
                    parameter_, ok = QInputDialog.getInt(self, f"Enter {parameter}", "Please enter a positive integer:")
                else:
                    parameter_, ok = QInputDialog.getDouble(self, f"Enter {parameter}", "Please enter a positive float:")

                if ok:
                    lineEdit.setText(str(parameter_))
                    continue  # Retry with the new parameter
                else:
                    return None  # Return None if user cancels
            else:
                break  # Valid parameter, exit loop
        return parameter_

    def validate_raduis(self, lineEdit):
        """
        Validates the radius entered by the user.

        This function prompts the user to enter a radius value and repeatedly asks for a new radius until a valid radius is entered. 
        The maximum allowed radius is 195. If the entered radius is greater than the maximum allowed radius, 
        an error message is displayed and the user is prompted to enter a different radius. 
        If the user cancels the input dialog, None is returned.

        Returns:
            int or None: The valid radius value entered by the user, or None if the user cancels the input dialog.
        """
        radius = None
        # Repeat until a valid radius is entered
        while True:
            try:
                radius = int(lineEdit.text())
                MAX_RADIUS = 195

                if radius > MAX_RADIUS or radius < 0:
                    raise ValueError
            except ValueError:
                self.show_error_message(f"Radius must be less than {MAX_RADIUS} and greater than 0.")
                # Prompt user to enter a different radius
                radius, ok = QInputDialog.getInt(self, "Enter Radius", f"Please enter a radius less than {MAX_RADIUS}:")
                if ok:
                    self.ui.radius_.setText(str(radius))
                    continue  # Retry with the new radius
                else:
                    return None  # Return None if user cancels
            else:
                break  # Valid radius, exit loop
        return radius
    
    def validate_center(self, lineEdit):
        """
        Validates the center entered by the user.

        This function prompts the user to enter a center value in the format "x, y" and 
        repeatedly asks for a new center until a valid format is entered. 
        If the entered center format is invalid, an error message is displayed 
        and the user is prompted to enter a different center. 
        If the user cancels the input dialog, None is returned.

        Returns:
            tuple or None: The valid center tuple entered by the user, or None if the user cancels the input dialog.
        """
        circle_center = None  

        while True:
            try:
                center_str = lineEdit.text()  # String of the form "x, y"

                # Check if the comma is present in the center string
                if ',' not in center_str:
                    raise ValueError
                
                # Split the string into two coordinates
                x, y = map(int, center_str.split(','))  # Parse the coordinates
                # Create a tuple of the coordinates
                circle_center = (x, y)  # Center of the circle

            except ValueError:
                self.show_error_message("Invalid center format. Please enter coordinates separated by a comma.")
                # Prompt user to enter a different center
                center_str, ok = QInputDialog.getText(self, "Enter Center", "Please enter center coordinates in the format x, y:")
                if ok:
                    lineEdit.setText(center_str)
                    continue  # Retry with the new center
                else:
                    return None  # Return None if user cancels
            else:
                break  # Valid center format, exit loop

        return circle_center

