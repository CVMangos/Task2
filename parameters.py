import random
import math

class Parameters:

    def __init__(self, points):
        self.points = points


    def is_inside_shape(self, x, y):
        """
        Checks if a point (x, y) is inside the shape using the ray-casting algorithm.
        
        Parameters:
            x (float): X-coordinate of the point.
            y (float): Y-coordinate of the point.
        
        Returns:
            bool: True if the point is inside the shape, False otherwise.
        """
        num_points = len(self.points)
        inside = False
        for i in range(num_points):
            xi, yi = self.points[i]
            xj, yj = self.points[(i + 1) % num_points]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        return inside


    def calculate_area_free_shape(self, bounding_box):
        """
        Calculates the area of a free-form shape using a random point generation method within a given bounding box.
        
        :param bounding_box: A tuple of four values representing the minimum and maximum x and y coordinates of the bounding box.
        :type bounding_box: tuple
        
        :return: The estimated area of the shape.
        :rtype: float
        """

        # The bounding box defines a rectangular region that completely contains the free-form shape.
        # Random points are generated uniformly within this bounding box.
        # For each generated point, it's determined whether it lies inside or outside the shape.
        # The ratio of points inside the shape to the total number of generated points is used to estimate the area of the shape.
        num_points_inside = 0
        num_points_total = 100000  # Adjust this value for desired accuracy
        
        min_x, max_x, min_y, max_y = bounding_box
        for _ in range(num_points_total):
            x = random.uniform(min_x, max_x)
            y = random.uniform(min_y, max_y)
            if self.is_inside_shape(x, y):
                num_points_inside += 1
        
        total_area = (max_x - min_x) * (max_y - min_y)
        shape_area = total_area * (num_points_inside / num_points_total)
        return shape_area


    def calculate_perimeter(self):
        """
        Calculates the perimeter of an irregular shape defined by a list of (x, y) points.
        
        Parameters:
        - None.
        
        Returns:
        - perimeter (float): Perimeter of the shape.
        """
        perimeter = 0
        num_points = len(self.points)
        for i in range(num_points):
            # Get the coordinates of the current and next point in the shape.
            x1, y1 = self.points[i]
            x2, y2 = self.points[(i + 1) % num_points]  # Wrap around to the first point creating a closed shape.
            # Calculate the distance between the two points.
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            # Add the distance to the total perimeter.
            perimeter += distance
        return perimeter
    

    def get_chain_code(self):
        """
        This function calculates the chain code for a closed polygon represented by a list of points.

        Args:
            points: A list of tuples representing the (x, y) coordinates of the polygon vertices. At least 3 points are required for a valid polygon.

        Returns:
            The chain code as a string, representing the directional sequence along the polygon's edges based on the provided directional scheme.
        """
        if len(self.points) < 3:
            raise ValueError("Polygon requires at least 3 points")

        chain_code = ""
        prev_x, prev_y = self.points[0]  # Store previous point coordinates

        for x, y in self.points[1:]:  # Iterate through points starting from the second element (assuming closed polygon)
            dx = x - prev_x  # Calculate difference in x-coordinate
            dy = y - prev_y  # Calculate difference in y-coordinate
            
            # Determine code based on directional movement between points
            code = 0
            if dx > 0:  # Movement to the right
                code = 0  # 0: Right
            elif dx < 0:  # Movement to the left
                code = 4  # 4: Left
            else:  # No horizontal movement
                if dy > 0:
                    code = 2  # 2: Up
                else:
                    code = 6  # 6: Down
            
            # Diagonal movements
            if abs(dx) == abs(dy):  # Check for diagonal movement
                if dx > 0 and dy > 0:
                    code = 1  # 1: Up-right
                elif dx > 0 and dy < 0:
                    code = 7  # 7: Down-right
                elif dx < 0 and dy > 0:
                    code = 3  # 3: Up-left  
                else:
                    code = 5  # 5: Down-left

            chain_code += str(code)  # Convert code to string and add to chain code
            prev_x, prev_y = x, y  # Update previous point for next iteration
        
        return chain_code
