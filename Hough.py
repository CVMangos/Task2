import numpy as np
import math
import cv2
from collections import defaultdict

class Hough:
    def __init__(self, original_img):
        self.image = original_img

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

    def canny_detector(self, low_threshold: float, high_threshold: float, smoothing_degree: float):
        # Step 2: Apply Gaussian blur
        blurred_image = cv2.GaussianBlur(cv2.convertScaleAbs(self.image), (3, 3), sigmaX= smoothing_degree, sigmaY= smoothing_degree)
        # Step 3: Compute gradient magnitude and direction
        gradient_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_direction = np.arctan2(gradient_y, gradient_x) * (180 / np.pi)

        # Step 4: Non-maximum suppression
        suppressed_image = np.zeros_like(gradient_magnitude)
        for i in range(1, gradient_magnitude.shape[0] - 1):
            for j in range(1, gradient_magnitude.shape[1] - 1):
                angle = gradient_direction[i, j]
                if (0 <= angle < 22.5) or (157.5 <= angle <= 180) or (-22.5 <= angle < 0) or (-180 <= angle < -157.5):
                    if (gradient_magnitude[i, j] >= gradient_magnitude[i, j - 1]) and \
                            (gradient_magnitude[i, j] >= gradient_magnitude[i, j + 1]):
                        suppressed_image[i, j] = gradient_magnitude[i, j]
                elif (22.5 <= angle < 67.5) or (-157.5 <= angle < -112.5):
                    if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j - 1]) and \
                            (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j + 1]):
                        suppressed_image[i, j] = gradient_magnitude[i, j]
                elif (67.5 <= angle < 112.5) or (-112.5 <= angle < -67.5):
                    if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j]) and \
                            (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j]):
                        suppressed_image[i, j] = gradient_magnitude[i, j]
                elif (112.5 <= angle < 157.5) or (-67.5 <= angle < -22.5):
                    if (gradient_magnitude[i, j] >= gradient_magnitude[i - 1, j + 1]) and \
                            (gradient_magnitude[i, j] >= gradient_magnitude[i + 1, j - 1]):
                        suppressed_image[i, j] = gradient_magnitude[i, j]

        # Step 5: Hysteresis thresholding
        edge_image = np.zeros_like(suppressed_image)
        weak_edges = (suppressed_image > low_threshold) & (suppressed_image <= high_threshold)
        strong_edges = suppressed_image > high_threshold
        edge_image[strong_edges] = 255
        for i in range(1, edge_image.shape[0] - 1):
            for j in range(1, edge_image.shape[1] - 1):
                if weak_edges[i, j]:
                    if np.any(strong_edges[i - 1:i + 2, j - 1:j + 2]):
                        edge_image[i, j] = 255

        return edge_image.astype(np.uint8)
    