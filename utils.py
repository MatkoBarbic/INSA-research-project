import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling


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


def create_and_save_psf(save_dir_path: str, grid: np.array, sigma_x: float, sigma_y: float, eta: float, mean_x: float = 0, mean_y: float = 0):
    '''
        Calculates normal point spread function for a 2D grid and save it to a desired directory.
        The PSF is saved in the format: {sigma_x}_{sigma_y}_{eta}.npy.

        Args:
            save_dir_path (str): Path to the directory in which the PSFs should be saved.
            grid (np.array[int]): The grid containing coordinate values for which the psf value should be calculated.
            sigma_x (float): Variation of the x axis of the 2D normal distribution.
            sigma_y (float): Variation of the y axis of the 2D normal distribution.
            eta (float): Rotation angle.
            mean_x (float): Mean of the x axis of the 2D normal distribution. Defaults to 0.
            mean_y (float): Mean of the y axis of the 2D normal distribution. Defaults to 0.

        Returns:
            None
    '''
    if not os.path.isdir(save_dir_path):
        os.makedirs(save_dir_path)
              
    psf = get_normal_psf_for_grid(grid=grid, sigma_x=sigma_x, sigma_y=sigma_y, eta=eta, mean_x=mean_x, mean_y=mean_y)
    np.save(os.path.join(save_dir_path, "{sigma_x}_{sigma_y}_{eta}.npy".format(sigma_x=sigma_x, sigma_y=sigma_y, eta=eta)), psf)


def load_psf(save_dir_path: str, sigma_x: float, sigma_y: float, eta: float):
    '''
        Loads a saved PSF based off of sigma_x, sigma_y and eta parameters.    
    
        Args:
            save_dir_path (str): Path to the directory in which the PSFs should be saved.
            sigma_x (float): Variation of the x axis of the 2D normal distribution.
            sigma_y (float): Variation of the y axis of the 2D normal distribution.
            eta (float): Rotation angle.

        Returns:
            (np.array[int]): Loaded PSF.
    '''
    psf = np.load(os.path.join(save_dir_path, "{sigma_x}_{sigma_y}_{eta}.npy".format(sigma_x=sigma_x, sigma_y=sigma_y, eta=eta)))
    return psf


def apply_psf(img: np.array, psf: np.array):
    '''
        Apply PSF to image.    
    
        Args:
            img (np.array[int]): The image to which the PSF should be applied.
            psf (np.array[int]): The point spread function.

        Returns:
            (np.array[int]): The imaged after appling the PSF.
    '''
    convolved_channels = []
    fft_psf = np.fft.fft2(psf)

    for channel in range(img.shape[-1]):
        fft_img = np.fft.fft2(img[:, :, channel])
        psf_img = np.abs(np.fft.ifftshift(np.fft.ifft2(fft_img * fft_psf)))
        convolved_channels.append(psf_img)

    psf_img = np.stack(convolved_channels, axis=-1)
    return psf_img.astype(np.uint8)


def downsample(img: np.array, factor: int):
    '''
        Downscample an image by a factor.

        Args:
            grid (np.array[int]): Image to be downscaled.
            factor (int): Factor by which the images is to be downsampled.

        Returns:
            (np.array[int]): Downsampled image.
    '''
    return img[::factor, ::factor]


def create_tiles(path_to_image: str, output_folder: str, tile_size: int, overlap: int = 0, save_metadata: bool = False):
    '''
    Create tiles from a large raster image.

    Args:
        path_to_image (str): Path to the input raster image.
        output_folder (str): Path to the output folder where tiles will be saved.
        tile_size (int): Size of each tile.
        overlap (int): Number of pixels to overlap between adjacent tiles.
        save_metadata (bool): Whether metadata should be saved along with the images. (TODO)

    Returns:
        None
    '''
    # Open the input raster image using rasterio
    with rasterio.open(path_to_image) as src:
        # Calculate the number of rows and columns for tiles
        num_rows = (src.height - overlap) // (tile_size - overlap)
        num_cols = (src.width - overlap) // (tile_size - overlap)

        # Iterate over rows and columns to create tiles
        for row in range(num_rows):
            for col in range(num_cols):
                # Define a window for each tile
                window = rasterio.windows.Window(
                    col * (tile_size - overlap),
                    row * (tile_size - overlap),
                    tile_size, tile_size
                )

                # Read the data for the tile
                tile_data = src.read(window=window)

                # Create a new dataset profile for the tile
                tile_profile = src.profile.copy()
                tile_profile.update({
                    'width': tile_size,
                    'height': tile_size,
                    'transform': src.window_transform(window),
                    'driver': 'JPEG',  # Specify the driver for JPEG format
                    'count': 3 if len(src.shape) == 3 else 1,  # 3 bands for RGB, adjust as needed
                    'dtype': 'uint8',  # Use uint8 for JPG images
                })

                if not os.path.isdir(output_folder):
                    os.makedirs(output_folder)

                # Define the output path for the tile
                output_path = os.path.join(output_folder, f'tile_{row}_{col}.jpg')

                # Write the tile data to a new JPEG file
                with rasterio.open(output_path, 'w', **tile_profile) as dst:
                    dst.write(tile_data.astype('uint8'))  # Ensure data type is uint8 for JPEG
