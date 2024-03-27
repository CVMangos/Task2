import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def load_img(file_name):
    img = cv2.imread(file_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img
    return cv2.resize(img, (500,500))

def initialize_snake(shape_center=(250,250), radius=200, N=300):
    """
    Initializes N points uniformly distributed along a circle.
    
    Parameters:
    - shape_center: A tuple of (x, y) coordinates for the center of the circle.
    - radius: The radius of the circle.
    - N: The number of points to generate along the circle.
    
    Returns:
    - points: An array of shape (N, 2), where each row is the (x, y) coordinate of a point.
    """
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)  # Equally spaced angle values
    x = shape_center[0] + radius * np.cos(theta)  # x coordinates of the points
    y = shape_center[1] + radius * np.sin(theta)  # y coordinates of the points
    
    return x,y

def create_A(a, b, N):
    """
    a: float
    alpha parameter

    b: float
    beta parameter

    N: int
    the number of points sampled on the snake curve
    """
    row = np.r_[
        -2*a - 6*b, 
        a + 4*b,
        -b,
        np.zeros(N-5),
        -b,
        a + 4*b
    ]
    A = np.zeros((N,N))
    for i in range(N):
        A[i] = np.roll(row, i)
    return A



def create_external_edge_force_gradients_from_img(img, sigma=20):
    """
    Given an image, returns 2 functions, fx & fy, that compute
    the gradient of the external edge force in the x and y directions.

    img: ndarray
        The image.
    """
    smoothed = cv2.GaussianBlur(img, (9, 9),sigma)
    d_x, d_y = np.gradient(smoothed)
    d_x = cv2.GaussianBlur(d_x, (9,9), sigma)
    d_y = cv2.GaussianBlur(d_y, (9,9), sigma)
    mag = (d_x**2 + d_y**2)**(0.5)
    plt.imshow(mag, cmap="gray")
    plt.show()
    dd_x, dd_y = np.gradient( mag-mag.min()/(mag.max()-mag.min()) )

    def fx(x, y):
        """
        Return external edge force in the x direction.

        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """

        x = np.clip(x, 0, img.shape[1]-1)
        y = np.clip(y, 0, img.shape[0]-1)

        return dd_x[ (y.round().astype(int), x.round().astype(int)) ]
    
    def fy(x, y):
        """
        Return external edge force in the y direction.

        x: ndarray
            numpy array of floats.
        y: ndarray:
            numpy array of floats.
        """

        x = np.clip(x, 0, img.shape[1]-1)
        y = np.clip(y, 0, img.shape[0]-1)

        return dd_y[ (y.round().astype(int), x.round().astype(int)) ]

    return fx, fy



def iterate_snake(x, y, a, b, fx, fy, gamma=0.1, n_iters=10, return_all=True):
    """
    x: ndarray
        intial x coordinates of the snake

    y: ndarray
        initial y coordinates of the snake

    a: float
        alpha parameter

    b: float
        beta parameter

    fx: callable
        partial derivative of first coordinate of external energy function. This is the first element of the gradient of the external energy.

    fy: callable
        see fx.

    gamma: float
        step size of the iteration
    
    n_iters: int
        number of times to iterate the snake

    return_all: bool
        if True, a list of (x,y) coords are returned corresponding to each iteration.
        if False, the (x,y) coords of the last iteration are returned.
    """
    A = create_A(a,b,x.shape[0])
    B = np.linalg.inv(np.eye(x.shape[0]) - gamma*A)
    if return_all:
        snakes = []

    for i in range(n_iters):
        x_ = np.dot(B, x + gamma*fx(x,y))
        y_ = np.dot(B, y + gamma*fy(x,y))
        x, y = x_.copy(), y_.copy()
        if return_all:
            snakes.append( (x_.copy(),y_.copy()) )

    if return_all:
        return snakes
    else:
        return (x,y)













# pth = "/home/mohamed/Documents/GitHub/Task2/snake/WhatsApp Image 2023-03-24 at 10.31.58 PM.jpeg"
# img = load_img(pth)

# alpha = 0.1*25
# beta = 100
# gamma = 2
# iterations = 100
# x,y = initialize_snake()
# # fx and fy are callable functions
# fx, fy = create_external_edge_force_gradients_from_img( img )
# xx,yy = iterate_snake(
#     x = x,
#     y = y,
#     a = alpha,
#     b = beta,
#     fx = fx,
#     fy = fy,
#     gamma = gamma,
#     n_iters = iterations,
#     return_all = False
# )
# plt.imshow(img)
# plt.plot(xx, yy, color='red')
# plt.show()
