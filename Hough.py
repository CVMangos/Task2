import numpy as np
import math
import cv2
from collections import defaultdict


class Hough:
    def __init__(self, original_img):
        self.image = original_img
        self.gradient_directions = None

    def detect_lines(self, low_threshold: float, high_threshold: float, smoothing_degree: float, rho: int, theta: float):
        image = self.image.copy()
        edges = self.canny_detector(low_threshold, high_threshold, smoothing_degree)
        threshold = 700
        diagonal = np.sqrt(image.shape[0]**2 + image.shape[1]**2)
        theta_angles = np.arange(-np.pi/2, np.pi/2, theta)
        rho_values = np.arange(-diagonal, diagonal, rho)
        num_thetas = len(theta_angles)
        num_rhos = len(rho_values)
        accumulator = np.zeros([num_rhos, num_thetas])
        sins = np.sin(theta_angles)
        coss = np.cos(theta_angles)
        xs, ys = np.where(edges > 0)
        for x,y in zip(xs,ys):
            for t in range(num_thetas):
                current_rho = x * coss[t] + y * sins[t]
                rho_pos = np.where(current_rho > rho_values)[0][-1]
                accumulator[rho_pos, t] += 1

        final_rho_index, final_theta_index = np.where(accumulator > threshold)
        final_rho = rho_values[final_rho_index]    
        final_theta = theta_angles[final_theta_index]

        polar_coordinates = np.vstack([final_rho, final_theta]).T 
        image_colored = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
 
        for r_theta in polar_coordinates:
            r, theta = r_theta
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*r
            y0 = b*r
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(image_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image_colored    
       

    def detect_circles(self, low_threshold: float, high_threshold: float, smoothing_degree: float, min_radius: float, max_radius: float, min_dist: float):
        image = cv2.cvtColor(self.image.copy(), cv2.COLOR_GRAY2BGR)
        edges = self.canny_detector(low_threshold, high_threshold, smoothing_degree)
        num_thetas = 10
        threshold = 6
        points = []
        for r in range(min_radius, max_radius + 1):
            for t in range(num_thetas):
                points.append((r, int(r * np.cos(2 * np.pi * t / num_thetas)), int(r * np.sin(2 * np.pi * t / num_thetas))))

        acc = defaultdict(int)
        edge_points = np.where(edges == 255)
        for x, y in zip(edge_points[1][::2], edge_points[0][::2]):
            for r, dx, dy in points:
                a = x - dx
                b = y - dy
                acc[(a, b, r)] += 1

        circles = []
        for k, v in sorted(acc.items(), key=lambda i: -i[1]):
            x, y, r = k
            if v  >= threshold and all(np.sqrt((x - xc) ** 2 + (y - yc) ** 2) >= min_dist for xc, yc, _ in circles):
                circles.append((x, y, r))
                cv2.circle(image, (x, y), r, (0, 255, 0), 2)
            
        return image

    def canny_detector(self, low_threshold=30, high_threshold=100, smoothing_degree=4):
        blurred_image = self.apply_gaussian_blur(smoothing_degree)
        gradient_magnitude, gradient_direction = self.compute_gradients(blurred_image)
        self.gradient_directions = gradient_direction.copy()
        suppressed_image = self.non_maximum_suppression(gradient_magnitude, gradient_direction)
        self.suppressed_image = suppressed_image
        edge_image = self.hysteresis_thresholding(suppressed_image, low_threshold, high_threshold)
        return edge_image.astype(np.uint8)

    def apply_gaussian_blur(self, smoothing_degree=4):
        return cv2.GaussianBlur(self.image, (5, 5), 4)

    def compute_gradients(self, image):
        gradient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)
        return gradient_magnitude, gradient_direction

    def non_maximum_suppression(self, gradient_magnitude, gradient_direction):
        suppressed_image = np.zeros_like(gradient_magnitude)
        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                angle = gradient_direction[i, j]
                if self.is_local_maximum(angle, gradient_magnitude, i, j):
                    suppressed_image[i, j] = gradient_magnitude[i, j]
        return suppressed_image

    def is_local_maximum(self, angle, gradient_magnitude, i, j):
        angle = self.normalize_angle(angle)
        if 0 <= angle < 22.5 or 157.5 <= angle <= 180 or -22.5 <= angle < 0 or -180 <= angle < -157.5:
            return gradient_magnitude[i, j] >= gradient_magnitude[i, j - 1] and \
                   gradient_magnitude[i, j] >= gradient_magnitude[i, j + 1]
        elif 22.5 <= angle < 67.5 or -157.5 <= angle < -112.5:
            return gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j - 1] and \
                   gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j + 1]
        elif 67.5 <= angle < 112.5 or -112.5 <= angle < -67.5:
            return gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j] and \
                   gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j]
        elif 112.5 <= angle < 157.5 or -67.5 <= angle < -22.5:
            return gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j + 1] and \
                   gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j - 1]
        return False

    def normalize_angle(self, angle):
        angle %= 180
        if angle < 0:
            angle += 180
        return angle

    def hysteresis_thresholding(self, suppressed_image, low_threshold=30, high_threshold=100):
        low_threshold = low_threshold
        high_threshold = high_threshold
        edge_image = np.zeros_like(suppressed_image)
        weak_edges = (suppressed_image > low_threshold) & (suppressed_image <= high_threshold)
        strong_edges = suppressed_image > high_threshold
        edge_image[strong_edges] = 255
        for i in range(1, edge_image.shape[0] - 1):
            for j in range(1, edge_image.shape[1] - 1):
                if weak_edges[i, j]:
                    if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                        edge_image[i, j] = 255
        return edge_image
    def hough_ellipse(self, low_threshold=30, high_threshold=100, smoothing_degree=5):
        # Image size
        rows, columns = self.image.shape[:2]
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = self.canny_detector(low_threshold, high_threshold, smoothing_degree)
        # Edges
        M, Ang = edges, self.gradient_directions

        # Accumulator

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
        threshold = 50
        overlap_threshold = .2
        # Threshold the accumulator to find ellipse centers
        ellipse_centers = np.argwhere(acc > threshold)

        # Extract ellipses using the centers
        ellipses = []
        for center in ellipse_centers:
            # Calculate major and minor axes lengths
            major_axis = 50 # Example value, replace with actual length
            minor_axis = 30  # Example value, replace with actual length

            # Assume angle is 0 (replace with actual angle if available)
            angle = 0

            # Construct ellipse parameters
            ellipse_params = ((center[1], center[0]), (major_axis, minor_axis), angle)

            # Append ellipse parameters to list
            ellipses.append(ellipse_params)

        # Apply non-maximum suppression
        filtered_ellipses = []
        for ellipse in ellipses:
            center, axes, angle = ellipse
            major_axis, minor_axis = axes
            is_max = True
            for other_ellipse in filtered_ellipses:
                other_center, other_axes, _ = other_ellipse
                overlap = self.calculate_overlap(center, axes, other_center, other_axes)
                if overlap > overlap_threshold:
                    if acc[center[1], center[0]] < acc[other_center[1], other_center[0]]:
                        is_max = False
                        break
                    else:
                        filtered_ellipses.remove(other_ellipse)
            if is_max:
                filtered_ellipses.append(ellipse)

        # Overlay ellipses on original image
        output_image = self.image.copy()
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
        for ellipse_params in filtered_ellipses:
            center, axes, angle = ellipse_params
            major_axis, minor_axis = axes
            cv2.ellipse(output_image, center, (major_axis, minor_axis), angle, 0, 360, (0, 255, 0), 2)  # Draw ellipse

        return output_image.astype(np.uint8)

    def calculate_overlap(self, center1, axes1, center2, axes2):
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