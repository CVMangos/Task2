import cv2
import numpy as np


class EdgeDetector:
    def __init__(self, original_img):
        self.gray = cv2.convertScaleAbs(original_img)
        self.gradient_direction = None
        self.suppressed_image = None

    def sobel_detector(self):
        # Define Sobel kernels
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        # Convolve the image with the kernels
        gradient_x = cv2.filter2D(self.gray, cv2.CV_64F, sobel_x)
        gradient_y = cv2.filter2D(self.gray, cv2.CV_64F, sobel_y)

        # Compute gradient magnitude
        gradient_magnitude = cv2.sqrt(cv2.pow(gradient_x, 2), cv2.pow(gradient_y, 2))

        gradient_magnitude *= 255.0 / gradient_magnitude.max()

        return gradient_magnitude.astype(np.uint8)

    def roberts_detector(self):
        kernel_x = np.array([[1, 0], [0, -1]])
        kernel_y = np.array([[0, 1], [-1, 0]])

        roberts_x = cv2.filter2D(self.gray, cv2.CV_64F, kernel_x)
        roberts_y = cv2.filter2D(self.gray, cv2.CV_64F, kernel_y)
        roberts = cv2.sqrt(cv2.pow(roberts_x, 2), cv2.pow(roberts_y, 2))

        return roberts.astype(np.uint8)

    def canny_detector(self):
        blurred_image = self.apply_gaussian_blur()
        gradient_magnitude, gradient_direction = self.compute_gradients(blurred_image)
        self.gradient_direction = gradient_direction
        suppressed_image = self.non_maximum_suppression(gradient_magnitude, gradient_direction)
        self.suppressed_image = suppressed_image
        edge_image = self.hysteresis_thresholding(suppressed_image)
        return edge_image.astype(np.uint8)

    def apply_gaussian_blur(self):
        return cv2.GaussianBlur(self.gray, (5, 5), 4)

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

    def hysteresis_thresholding(self, suppressed_image):
        low_threshold = 30
        high_threshold = 100
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
    def prewitt_detector(self):
        kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        prewitt_x = cv2.filter2D(self.gray, cv2.CV_64F, kernel_x)
        prewitt_y = cv2.filter2D(self.gray, cv2.CV_64F, kernel_y)
        prewitt = cv2.sqrt(cv2.pow(prewitt_x, 2), cv2.pow(prewitt_y, 2))

        return prewitt.astype(np.uint8)
