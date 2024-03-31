import numpy as np
import cv2
from math import sqrt, pow


def convert_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def calculate_internal_energy(point, previous_point, next_point, alpha):
    dx1 = point[0] - previous_point[0]
    dy1 = point[1] - previous_point[1]
    dx2 = next_point[0] - point[0]
    dy2 = next_point[1] - point[1]
    denominator = pow(dx1 * dx1 + dy1 * dy1, 1.5)
    curvature = 0 if denominator == 0 else (dx1 * dy2 - dx2 * dy1) / denominator
    return alpha * curvature


def calculate_external_energy(image, point, beta):
    return -beta * image[point[1], point[0]]


def calculate_gradients(point, prev_point, gamma):
    dx = point[0] - prev_point[0]
    dy = point[1] - prev_point[1]
    return gamma * (dx * dx + dy * dy)


def calculate_point_energy(image, point, prev_point, next_point, alpha, beta, gamma):
    internal_energy = calculate_internal_energy(point, prev_point, next_point, alpha)
    external_energy = calculate_external_energy(image, point, beta)
    gradients = calculate_gradients(point, prev_point, gamma)
    return internal_energy + external_energy + gradients


def snake_operation(image, curve, window_size, alpha, beta, gamma):
    new_curve = []
    window_index = (window_size - 1) // 2
    num_points = len(curve)

    for i in range(num_points):
        pt = curve[i]
        prev_pt = curve[(i - 1 + num_points) % num_points]
        next_pt = curve[(i + 1) % num_points]
        min_energy = float("inf")
        new_pt = pt

        for dx in range(-window_index, window_index + 1):
            for dy in range(-window_index, window_index + 1):
                move_pt = (pt[0] + dx, pt[1] + dy)
                energy = calculate_point_energy(
                    image, move_pt, prev_pt, next_pt, alpha, beta, gamma
                )
                if energy < min_energy:
                    min_energy = energy
                    new_pt = move_pt
        new_curve.append(new_pt)

    return new_curve


def initialize_contours(center, radius, number_of_points):
    print("initializing contours")
    curve = []
    current_angle = 0
    resolution = 360 / number_of_points

    for _ in range(number_of_points):
        x_p = int(center[0] + radius * np.cos(np.radians(current_angle)))
        y_p = int(center[1] + radius * np.sin(np.radians(current_angle)))
        current_angle += resolution
        curve.append((x_p, y_p))

    return curve


def draw_contours(image, snake_points):
    print("drawing contours")
    output_image = image.copy()
    for i in range(len(snake_points)):
        cv2.circle(output_image, snake_points[i], 4, (0, 0, 255), -1)
        if i > 0:
            cv2.line(output_image, snake_points[i - 1], snake_points[i], (255, 0, 0), 2)
    cv2.line(output_image, snake_points[0], snake_points[-1], (255, 0, 0), 2)
    return output_image


def active_contour(
    input_image,
    center,
    radius,
    num_iterations,
    num_points,
    window_size,
    alpha,
    beta,
    gamma,
):
    print("starting active contour")
    curve = initialize_contours(center, radius, num_points)
    # gray_image = convert_to_gray(input_image)
    gray_image = input_image

    for _ in range(num_iterations):
        curve = snake_operation(gray_image, curve, window_size, alpha, beta, gamma)

    output_image = draw_contours(input_image, curve)
    return curve, output_image


def active_contour_from_circle(
    input_image,
    circle_center,
    circle_radius,
    num_iterations,
    num_points,
    window_size,
    alpha,
    beta,
    gamma,
):
    print("starting active contour")
    snake_curve, output_image = active_contour(
        input_image,
        circle_center,
        circle_radius,
        num_iterations,
        num_points,
        window_size,
        alpha,
        beta,
        gamma,
    )
    return snake_curve, output_image


# Example usage
# input_image = cv2.imread("coin-png-500x501_5e76a44c_transparent_202166.png.png")
# output_image = input_image.copy()
#
# # Draw circle on the original image
# circle_center = input_image.shape[0]//2 , input_image.shape[1]//2
# circle_radius = 200  # Example radius
# cv2.circle(output_image, circle_center, circle_radius, (0, 255, 0), 2)
#
# cv2.imshow("Original Image with Circle", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
# num_iterations = 300
# num_points = 300
# window_size = 4
# alpha = 10
# beta = 3
# gamma = 1
#
# print("before")
# snake_curve, output_image = active_contour_from_circle(
#     input_image,
#     circle_center,
#     circle_radius,
#     num_iterations,
#     num_points,
#     window_size,
#     alpha,
#     beta,
#     gamma,
# )
#
# cv2.imshow("Output Image", output_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

