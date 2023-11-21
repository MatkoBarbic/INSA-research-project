import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px

def normal_psf(input_coordinates: list, sigma_x: float, sigma_y: float, eta: float, mean_x: float = 0, mean_y: float = 0):
    '''
        Calculates normal point spread function for a 2D coordinates input.

        Args:
            input_coordinates (list[int]): The image coordinates for which the psf value should be calculated.
            sigma_x (float): Variation of the x axis of the 2D normal distribution.
            sigma_y (float): Variation of the y axis of the 2D normal distribution.
            eta (float): Rotation angle.
            mean_x (float): Mean of the x axis of the 2D normal distribution. Defaults to 0.
            mean_y (float): Mean of the y axis of the 2D normal distribution. Defaults to 0.
        
        Returns:
            (float): Value of psf.
    '''
    input_coordinates = np.array(input_coordinates).reshape(-1, 1)
    mean = np.array([mean_x, mean_y]).reshape(-1, 1)
    sigma = np.array([[sigma_x, 0], [0, sigma_y]])
    angle_matrix = np.array([[math.cos(eta), math.sin(eta)], [-math.sin(eta), math.cos(eta)]])
    cov_matrix = angle_matrix @ sigma @ angle_matrix.T
    inv_cov = np.linalg.inv(cov_matrix)

    psf = 1/(2 * math.pi * math.sqrt(sigma_x*sigma_y)) * float(np.exp(-1/2 * (input_coordinates - mean).T @ inv_cov @ (input_coordinates - mean)))
    return psf

def get_normal_psf_for_grid(grid: np.array, sigma_x: float, sigma_y: float, eta: float, mean_x: float = 0, mean_y: float = 0):
    '''
        Calculates normal point spread function for a 2D grid.

        Args:
            grid (np.array[int]): The grid containing coordinate values for which the psf value should be calculated.
            sigma_x (float): Variation of the x axis of the 2D normal distribution.
            sigma_y (float): Variation of the y axis of the 2D normal distribution.
            eta (float): Rotation angle.
            mean_x (float): Mean of the x axis of the 2D normal distribution. Defaults to 0.
            mean_y (float): Mean of the y axis of the 2D normal distribution. Defaults to 0.

        Returns:
            (np.array[float]): Value of psf for the whole grid.
    '''

    psf = []
    for i in range(len(grid)):
        psf_row = []
        for j in range(len(grid)):
            psf_row.append(normal_psf(grid[i, j], sigma_x=sigma_x, sigma_y=sigma_y, eta=eta, mean_x=mean_x, mean_y=mean_y))
        psf.append(psf_row)

    psf = np.array(psf)

    return psf


def update_psf(plot, grid: np.array, sigma_x: float, sigma_y: float, eta: float, mean_x: float, mean_y: float):
    '''
        Updates the PSF plot.

        Args:
            plot: Plot to be updated.
            grid (np.array[int]): The grid containing coordinate values for which the psf value should be calculated.
            sigma_x (float): Variation of the x axis of the 2D normal distribution.
            sigma_y (float): Variation of the y axis of the 2D normal distribution.
            eta (float): Rotation angle.
            mean_x (float): Mean of the x axis of the 2D normal distribution.
            mean_y (float): Mean of the y axis of the 2D normal distribution.

        Returns:
            (): Updated plot.
    '''
    new_psf = get_normal_psf_for_grid(grid=grid, sigma_x=sigma_x, sigma_y=sigma_y, eta=eta, mean_x=mean_x, mean_y=mean_y)
    new_psf = px.imshow(new_psf, color_continuous_scale='gray')
    new_psf.update_layout(coloraxis_showscale=False)

    new_psf = go.FigureWidget(new_psf)
    plot.data[0].z = new_psf.data[0].z


    