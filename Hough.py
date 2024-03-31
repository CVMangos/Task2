from Edge_Detector import EdgeDetector
import matplotlib.pyplot as plt

import cv2
import numpy as np

def traverse_edges(center, gradient_directions, gradient_magnitudes):
    rows, cols = gradient_directions.shape
    x, y = center
    edge_points = []

    for i in range(5):
        x1, y1 = x, y
        angle1 = gradient_directions[y, x]
        step_size = 1

        while True:
            dx = round(step_size * np.cos(angle1))
            dy = round(step_size * np.sin(angle1))
            x2, y2 = x1 + dx, y1 + dy

            if 0 <= x2 < cols and 0 <= y2 < rows:
                angle2 = gradient_directions[y2, x2]
                if abs(angle1 - angle2) > 0.1:
                    edge_points.append((x2, y2))
                    break
                else:
                    x1, y1 = x2, y2
                    step_size = gradient_magnitudes[y1, x1]  # Adjust step size based on gradient magnitude
            else:
                break

    return edge_points


def hough_ellipse(input_image):
    # Image size
    rows, columns = input_image.shape[:2]
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    detector = EdgeDetector(input_image_gray)
    edges = detector.canny_detector()
    # Edges
    M, Ang = edges, detector.gradient_direction

    # Accumulator
    acc = np.zeros((rows, columns))

    # Gather evidence
    for x1 in range(columns):
        for y1 in range(rows):
            if M[y1, x1] != 0:
                for i in range(60, 61):
                    x2 = x1 - i
                    y2 = y1 - i
                    incx = 1
                    incy = 0
                    for k in range(0, 8 * i):
                        if 0 < x2 < columns and 0 < y2 < rows:
                            if M[y2, x2] != 0:
                                m1 = Ang[y1, x1]
                                m2 = Ang[y2, x2]
                                if abs(m1 - m2) > 0.2:
                                    xm = (x1 + x2) / 2
                                    ym = (y1 + y2) / 2
                                    m1 = np.tan(m1)
                                    m2 = np.tan(m2)
                                    A = y1 - y2
                                    B = x2 - x1
                                    C = m1 + m2
                                    D = m1 * m2
                                    N = (2 * A + B * C)
                                    if N != 0:
                                        m = (A * C + 2 * B * D) / N
                                    else:
                                        m = 99999999
                                    if -1 < m < 1:
                                        for x0 in range(columns):
                                            y0 = round(ym + m * (x0 - xm))
                                            if 0 < y0 < rows:
                                                acc[y0, x0] += 1
                                    else:
                                        for y0 in range(rows):
                                            x0 = round(xm + (y0 - ym) / m)
                                            if 0 < x0 < columns:
                                                acc[y0, x0] += 1
                        x2 += incx
                        y2 += incy
                        if x2 > x1 + i:
                            x2 = x1 + i
                            incx = 0
                            incy = 1
                            y2 += incy
                        if y2 > y1 + i:
                            y2 = y1 + i
                            incx = -1
                            incy = 0
                            x2 += incx
                        if x2 < x1 - i:
                            x2 = x1 - i
                            incx = 0
                            incy = -1
                            y2 += incy

    # Threshold the accumulator to find ellipse centers
    threshold = 40
    overlap_threshold = .9
    # Threshold the accumulator to find ellipse centers
    ellipse_centers = np.argwhere(acc > threshold)

    # Extract ellipses using the centers
    ellipses = []
    for center in ellipse_centers:
        # Estimate major and minor axes lengths
        major_axis, minor_axis = estimate_axes_lengths(center, detector.gradient_direction, edges, threshold=100)

        # Construct ellipse parameters
        ellipse_params = ((center[1], center[0]), (major_axis, minor_axis), 0)  # Assume angle is 0

        # Append ellipse parameters to list
        ellipses.append(ellipse_params)

    # Apply non-maximum suppression
    filtered_ellipses = []
    for ellipse in ellipses:
        center, axes, _ = ellipse
        major_axis, minor_axis = axes
        is_max = True
        for other_ellipse in filtered_ellipses:
            other_center, other_axes, _ = other_ellipse
            overlap = calculate_overlap(center, axes, other_center, other_axes)
            if overlap > overlap_threshold:
                if acc[center[1], center[0]] < acc[other_center[1], other_center[0]]:
                    is_max = False
                    break
                else:
                    filtered_ellipses.remove(other_ellipse)
        if is_max:
            filtered_ellipses.append(ellipse)

    # Overlay ellipses on original image
    output_image = input_image.copy()
    for ellipse_params in filtered_ellipses:
        center, axes, angle = ellipse_params
        major_axis, minor_axis = axes
        cv2.ellipse(output_image, center, (major_axis, minor_axis), angle, 0, 360, (0, 255, 0), 2)  # Draw ellipse

    return output_image

def calculate_overlap(center1, axes1, center2, axes2):
    # Calculate distances between centers
    dx = center1[0] - center2[0]
    dy = center1[1] - center2[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)

    # Calculate maximum radius for each ellipse
    radius1 = max(axes1)
    radius2 = max(axes2)

    # Calculate overlap ratio
    overlap_ratio = min(radius1, radius2) / distance

    return overlap_ratio

def estimate_axes_lengths(center, gradient_directions, gradient_magnitudes, threshold=20):
    rows, cols = gradient_directions.shape
    neighbor_radius = 10  # Define the radius to search for neighboring points
    neighbor_values = []

    # Traverse edges from the center and collect gradient magnitudes
    edge_points = traverse_edges(center, gradient_directions, gradient_magnitudes)
    print(len(edge_points))
    for edge_point in edge_points:
        x, y = edge_point
        if 0 <= x < cols and 0 <= y < rows:
            neighbor_values.append(gradient_magnitudes[y, x])

    # Apply thresholding to select candidate points
    candidate_values = [v for v in neighbor_values if v >= threshold]
    print(len(candidate_values))
    # Estimate axes lengths based on candidate points
    if candidate_values:
        major_axis = np.mean(candidate_values)  # Example: Use mean of accumulator values
        minor_axis = np.median(candidate_values)  # Example: Use median of accumulator values
    else:
        # If no candidate points found, set default values
        major_axis = 100  # Example value, replace with appropriate default
        minor_axis = 30  # Example value, replace with appropriate default

    return int(major_axis), int(minor_axis)


# input_image = cv2.imread('Images/ellipses.png')  # Replace 'your_image_path.jpg' with the actual path to your image
#
# # Apply the hough_ellipse function
# output_image = hough_ellipse(input_image)
#
# # Display the original and resulting images side by side
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
# axes[0].set_title('Original Image')
# axes[0].axis('off')
# axes[1].imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
# axes[1].set_title('Resulting Image with Ellipses')
# axes[1].axis('off')
# plt.show()
