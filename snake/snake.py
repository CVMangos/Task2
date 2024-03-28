import numpy as np
import cv2
import matplotlib.pyplot as plt

class SnakeModel:
    def __init__(self, image, Xs, Ys):
        self.image = image
        self.x_vals = Xs
        self.y_vals = Ys

    def create_A_matrix(self, alpha, beta, num_points):
        """Create the A matrix for the Snake Iteration

        Parameters
        ----------
        alpha : float
            The stiffness parameter of the Snake Iteration
        beta : float
            The regularization parameter of the Snake Iteration
        num_points : int
            The number of points in the snake

        Returns
        -------
        A : array
            The A matrix to be used in the Snake Iteration
        """
        # Create the A matrix for the Snake Iteration
        #
        # The A matrix is a tridiagonal matrix with the following pattern:
        # [[-2*alpha - 6*beta   0   0   ...   0]
        #  [ alpha + 4*beta -beta   0   ...   0]
        #  [   0        -beta   0   0   ...   0]
        #  [   0           0   0   0   ...   0]
        #  [   ...          ... ... ... ... ...]
        #  [   0           0   0   0  -beta alpha + 4*beta]]
        #
        # where alpha is the stiffness parameter, beta is the regularization
        # parameter and num_points is the number of points in the snake
        row = np.r_[-2*alpha - 6*beta, alpha + 4*beta, -beta, np.zeros(num_points-5), -beta, alpha + 4*beta]
        A = np.zeros((num_points, num_points))
        for i in range(num_points):
            A[i] = np.roll(row, i)
        return A

    def create_external_edge_force_gradients(self, sigma=20):
        smoothed = cv2.GaussianBlur(self.image, (9, 9), sigma)
        d_x, d_y = np.gradient(smoothed)
        d_x = cv2.GaussianBlur(d_x, (9, 9), sigma)
        d_y = cv2.GaussianBlur(d_y, (9, 9), sigma)
        mag = (d_x**2 + d_y**2)**0.5
        dd_x, dd_y = np.gradient(mag - mag.min() / (mag.max() - mag.min()))

        def fx(x, y):
            x = np.clip(x, 0, self.image.shape[1] - 1)
            y = np.clip(y, 0, self.image.shape[0] - 1)
            return dd_x[(y.round().astype(int), x.round().astype(int))]

        def fy(x, y):
            x = np.clip(x, 0, self.image.shape[1] - 1)
            y = np.clip(y, 0, self.image.shape[0] - 1)
            return dd_y[(y.round().astype(int), x.round().astype(int))]

        return fx, fy

    def iterate_snake(self, alpha, beta, gamma=0.1, n_iters=10):
        fx, fy = self.create_external_edge_force_gradients()
        x = np.array(self.x_vals)
        y = np.array(self.y_vals)
        A = self.create_A_matrix(alpha, beta, x.shape[0])
        B = np.linalg.inv(np.eye(x.shape[0]) - gamma * A)

        for _ in range(n_iters):
            x_ = np.dot(B, x + gamma * fx(x, y))
            y_ = np.dot(B, y + gamma * fy(x, y))
            x, y = x_.copy(), y_.copy()
        self.x_vals, self.y_vals = x, y
        return x, y
    
    def add_snake_to_image(self):
        image_with_snake = self.image.copy()
        pts = np.array([(int(x), int(y)) for x, y in zip(self.x_vals, self.y_vals)], np.int32)
        # pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image_with_snake, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        return image_with_snake


# # Example usage
# image_path = "coin-png-500x501_5e76a44c_transparent_202166.png.png"
# snake_model = SnakeModel(image_path)
# alpha = 0.1 * 25
# beta = 100
# gamma = 2
# iterations = 100
# x, y = snake_model.x_vals, snake_model.y_vals
# fx, fy = snake_model.create_external_edge_force_gradients()
# xx, yy = snake_model.iterate_snake(x, y, alpha, beta, fx, fy, gamma, iterations, return_all=False)

# plt.imshow(snake_model.image, cmap='gray')
# plt.plot(xx, yy, color='red')
# plt.show()
