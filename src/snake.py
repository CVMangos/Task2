import numpy as np
import cv2
from math import pow

class SnakeContour:
    def __init__(self, image, circle_center, circle_radius, 
                 window_size=3, num_iterations=300, num_points=300, alpha=0.1, beta=100, gamma=0.1):
        self.original_img = image
        self.image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        self.circle_center = circle_center
        self.circle_radius = circle_radius
        self.window_size = window_size
        self.num_iterations = num_iterations
        self.num_points = num_points
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma


    def calculate_internal_energy(self, point, previous_point, next_point):
        """
        Calculate the internal energy of a point based on its neighbors

        Arguments:
            point {tuple} -- Coordinates of the point to calculate energy for
            previous_point {tuple} -- Coordinates of the previous point in the contour
            next_point {tuple} -- Coordinates of the next point in the contour

        Returns:
            float -- Internal energy of the point
        """
        # Calculate the difference of the coordinates of the point and its neighbors
        dx1 = point[0] - previous_point[0]
        dy1 = point[1] - previous_point[1]
        dx2 = next_point[0] - point[0]
        dy2 = next_point[1] - point[1]

        # Calculate the denominator of the curvature formula
        denominator = pow(dx1 * dx1 + dy1 * dy1, 1.5)

        # Calculate the curvature of the point
        curvature = 0 if denominator == 0 else (dx1 * dy2 - dx2 * dy1) / denominator

        # Return the internal energy of the point based on the curvature
        return self.alpha * curvature


    def calculate_external_energy(self, image, point):
        """
        Calculate the external energy of a point based on its intensity in the image

        Arguments:
            image {numpy.ndarray} -- Grayscale image
            point {tuple} -- Coordinates of the point to calculate energy for

        Returns:
            float -- External energy of the point
        """
        height, width = image.shape[:2]
        x, y = point[0], point[1]

        # Check if the point is within the image bounds
        if 0 <= x < width and 0 <= y < height:
            # Return the negative of the intensity of the point, which is the external energy
            return -self.beta * image[y, x]
        else:
            # Handle out-of-bounds points (e.g., by returning a default value)
            return 0  # Default to a value of 0, which has no effect on the contour



    def calculate_gradients(self, point, prev_point):
        """
        Calculate the gradients of the contour at a point based on the change in coordinates between the point and its neighbor

        Arguments:
            point {tuple} -- Coordinates of the point to calculate gradients for
            prev_point {tuple} -- Coordinates of the previous point in the contour

        Returns:
            float -- Gradients of the contour at the point
        """
        dx = point[0] - prev_point[0]
        dy = point[1] - prev_point[1]
        return self.gamma * (dx * dx + dy * dy)



    def calculate_point_energy(self, image, point, prev_point, next_point):
        """
        Calculate the energy of a point in the contour based on the internal energy, external energy, and gradients
        https://en.wikipedia.org/wiki/Active_contour_model

        Arguments:
            image {numpy.ndarray} -- Grayscale image
            point {tuple} -- Coordinates of the point to calculate energy for
            prev_point {tuple} -- Coordinates of the previous point in the contour
            next_point {tuple} -- Coordinates of the next point in the contour

        Returns:
            float -- Total energy of the point in the contour
        """
        internal_energy = self.calculate_internal_energy(point, prev_point, next_point)
        external_energy = self.calculate_external_energy(image, point)
        gradients = self.calculate_gradients(point, prev_point)
        return internal_energy + external_energy + gradients


    def snake_operation(self, image, curve):
        """
        Perform one iteration of the active contour method

        Arguments:
            image {numpy.ndarray} -- Grayscale image
            curve {list} -- Current curve in the contour

        Returns:
            list -- New curve in the contour
        """
        new_curve = []
        window_index = (self.window_size - 1) // 2  # Half of the window size - 1
        num_points = len(curve)

        for i in range(num_points):
            # Point, previous point, and next point in the contour
            pt = curve[i]
            prev_pt = curve[(i - 1 + num_points) % num_points]
            next_pt = curve[(i + 1) % num_points]

            # Initialize minimum energy and new point to current point
            min_energy = float("inf")
            new_pt = pt

            # Loop through window and calculate energy of each point
            for dx in range(-window_index, window_index + 1):
                for dy in range(-window_index, window_index + 1):
                    move_pt = (pt[0] + dx, pt[1] + dy)  # Calculate moved point
                    energy = self.calculate_point_energy(image, move_pt, prev_pt, next_pt)  # Calculate energy
                    if energy < min_energy:  # If the energy is lower than the current minimum
                        min_energy = energy  # Set the new minimum energy
                        new_pt = move_pt  # Set the new point

            new_curve.append(new_pt)  # Add new point to new curve

        return new_curve  # Return the new curve


    def initialize_contours(self):
        """
        Initializes a contour around the circle with the specified number of points

        Returns:
            list: Curve (a list of (x, y) tuples)
        """
        print("initializing contours")
        curve = []  # Initialize empty curve
        current_angle = 0  # Start at 0 degrees
        resolution = 360 / self.num_points  # Distance between points in degrees

        for _ in range(self.num_points):
            # Calculate point coordinates
            x_p = int(self.circle_center[0] + self.circle_radius * np.cos(np.radians(current_angle)))
            y_p = int(self.circle_center[1] + self.circle_radius * np.sin(np.radians(current_angle)))

            # Add point to curve and increment angle
            curve.append((x_p, y_p))
            current_angle += resolution

        return curve


    def draw_contours(self, image, snake_points):
        """
        Draws the contours of the snake on the input image

        Args:
            image (numpy array): Input image
            snake_points (list): List of (x, y) tuples representing the contour points

        Returns:
            numpy array: Output image with contours drawn
        """
        print("drawing contours")
        output_image = image.copy()

        # Loop through each point in the contour and draw it as a circle
        for i in range(len(snake_points)):
            cv2.circle(output_image, snake_points[i], 4, (0, 0, 255), -1)

            # If this isn't the first point, draw a line from the previous point to this one
            if i > 0:
                cv2.line(output_image, snake_points[i - 1], snake_points[i], (255, 0, 0), 2)

        # Draw a line from the first point to the last point to close the contour
        cv2.line(output_image, snake_points[0], snake_points[-1], (255, 0, 0), 2)

        return output_image


    def active_contour(self):
        """
        Runs active contour on the input image

        Returns:
            tuple: (curve, output_image) where curve is a list of (x, y) tuples representing
                the contour points and output_image is the image with the contour drawn
        """
        print("starting active contour")

        # Initialize the contour with a certain number of points
        curve = self.initialize_contours()

        # Run the snake operation a certain number of times
        for _ in range(self.num_iterations):
            curve = self.snake_operation(self.image, curve)

        # Draw the contour on the original image and save the output
        output_image = self.draw_contours(self.original_img, curve)
        cv2.imwrite("snake.png", output_image)

        return curve, output_image


