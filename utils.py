import numpy as np
import math
import plotly.graph_objects as go
import plotly.express as px
import os
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import cv2
import pandas as pd
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity
from skimage.metrics import mean_squared_error
from tqdm import tqdm
import torch

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
    return psf_img.astype(np.uint16)


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
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)

    img_name = path_to_image.split("/")[-1].split(".")[0]

    processed_images_list = list(set([processed_img_name.split("_tile_")[0] for processed_img_name in os.listdir(output_folder)]))

    if img_name not in processed_images_list:
        try:
            with rasterio.open(path_to_image) as src:
                num_rows = (src.height - overlap) // (tile_size - overlap)
                num_cols = (src.width - overlap) // (tile_size - overlap)
                
                # color_code = color_coding.query("image == @image_name")["color_coding"].iloc[0]
                
                for row in range(num_rows):
                    for col in range(num_cols):
                        window = rasterio.windows.Window(
                            col * (tile_size - overlap),
                            row * (tile_size - overlap),
                            tile_size, tile_size
                        )

                        tile_data = src.read(window=window, indexes=(1, 2, 3))

                        tile_profile = src.profile.copy()
                        tile_profile.update({
                            'width': tile_size,
                            'height': tile_size,
                            'transform': src.window_transform(window),
                            'driver': 'PNG',
                            'count': 3 if src.count == 3 or src.count == 4 else 1,
                            'dtype': 'uint16',
                        })

                        output_path = os.path.join(output_folder, f'{img_name}_tile_{row}_{col}.png')

                        if np.mean(tile_data) != 0:
                            with rasterio.open(output_path, 'w', **tile_profile) as dst:
                                dst.write(tile_data.astype('uint16'))
        except:
            print(img_name)

def process_pansharp_img(path_to_image: str, psf: np.array, downscale_factor: int = 5):
    '''
        A function that preprocesses pansharpened image data.
        
        The pipeline consists of:
            1. Applying the PSF to the image.
            2. Downscaling the image.
            3. Creating a grayscale image.
            4. Creating a low-resolution RGB image.
        
        Args:
            path_to_image (str): Path to original pansharpened data.
            psf (np.array[int]): The point spread function.
            downscale_factor (int): Downscaling factor for the panchromatic image.
        
        Returns:
            pan_img (np.array[int]): Downscaled panchromatic image.
            rgb_img (np.array[int]): Downscaled RGB image
            sharp_img (np.array[int]): Pansharpened image.
    '''
    sharp_img = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED)
    sharp_img = apply_psf(img=sharp_img, psf=psf)

    rgb_img = downsample(img=sharp_img, factor=downscale_factor * 2)

    sharp_img = downsample(img=sharp_img, factor=downscale_factor)
    pan_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2GRAY)

    rgb_img = cv2.resize(rgb_img, (pan_img.shape))

    return pan_img, rgb_img, sharp_img


def data_augmentation_pipeline(path_to_data: str, psf: np.array, path_to_pansharp: str, path_to_rgb: str, path_to_panchrom: str, downscale_factor: int = 5):
    '''
        A function that prepares the data for training.
        
        The pipeline consists of:
            1. Reading all the data in the folder and filtering non-image data.
            2. Applying the PSF to the image.
            3. Downscaling the image.
            4. Creating a grayscale image.
            5. Creating a low-resolution RGB image.
            6. Saving all the images into designated folders.
        
        Args:
            path_to_data (str): Path to original pansharpened data.
            psf (np.array[int]): The point spread function.
            path_to_pansharp (str): Path where the pansharpened data is to be saved.
            path_to_rgb (str): Path where the rgb data is to be saved.
            path_to_panchrom (str): Path where the panchromating data is to be saved.
            downscale_factor (int): Downscaling factor for the panchromatic image.
        
        Returns:
            None
    '''
    if not os.path.isdir(path_to_pansharp):
        os.makedirs(path_to_pansharp)

    # if not os.path.isdir(path_to_rgb):
    #     os.makedirs(path_to_rgb)

    # if not os.path.isdir(path_to_panchrom):
    #     os.makedirs(path_to_panchrom)

    processed_images = os.listdir(path_to_pansharp)

    for img_path in tqdm(os.listdir(path_to_data)):
        if img_path[-3:] != "xml" and img_path not in processed_images:
            full_path_to_img = os.path.join(path_to_data, img_path)
            pan_img, rgb_img, sharp_img = process_pansharp_img(path_to_image=full_path_to_img, psf=psf, downscale_factor=downscale_factor)

            cv2.imwrite(os.path.join(path_to_pansharp, img_path), sharp_img)
            # cv2.imwrite(os.path.join(path_to_rgb, img_path), rgb_img)
            # cv2.imwrite(os.path.join(path_to_panchrom, img_path), pan_img)


def get_pansharpening_scores(img1: np.array, img2: np.array):
    '''
        A function that calculates the simalarity between two images.

        The scores are calculated by:
            1. Pearson correlation
            2. Structural similarity (SSIM)
            3. Mean squared error (MSE)

        Args:
            img1 (np.array[int]): The first image.
            img2 (np.array[int]): The second image.
        
        Returns:
            (pd.DataFrame): The DataFrame in a form of a pandas Series.
    '''
    scores = pd.Series(name="Score")

    img1[np.isnan(img1)] = 0
    img2[np.isnan(img2)] = 0

    img1_flat = img1.reshape(-1, 3)
    img2_flat = img2.reshape(-1, 3)

    pearson_scores = []
    for band in range(3):
        band_score = pearsonr(img1_flat[:, band], img2_flat[:, band])[0]
        pearson_scores.append(band_score)

    scores["Pearson"] = np.mean(pearson_scores)
    scores["SSIM"] = structural_similarity(img1, img2, channel_axis=2, data_range=1)
    scores["MSE"] = mean_squared_error(img1, img2)

    return scores.to_frame()


def evaluate_pansharpening(path_to_images: str, algorithm: callable, train_perc: float, train: bool, downsample_rgb: int = 2):
    '''
        A function that evaluates a classical pansharpening alhorithm on all scores.

        Args:
            path_to_images (str): Path to the dataset on which the algorithms should be evaluated.
            algorithm (callable): Pansharpening algorithm that will be evaluated.
            train_perc (float): A float between 0 and 1 that represents the percentage of the dataset that is used as training data in deep learning methods.
            train (bool): Should the algorithms be evaluated on the train dataset.
            downsample_rgb (int): Factor by which the rgb image will be degraded.
        
        Returns:
            (pd.DataFrame): A pandas DataFrame containing the scores.
    '''
    all_scores = []

    image_paths = sorted(os.listdir(path_to_images))
    if train:
        image_paths = image_paths[:int(len(image_paths) * train_perc)]
    else:
        image_paths = image_paths[int(len(image_paths) * train_perc): ]

    for image_name in tqdm(image_paths):
        img = cv2.imread(os.path.join(path_to_images, image_name), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        rgb_img = downsample(img=img, factor=downsample_rgb)
        pan_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        rgb_img = cv2.resize(rgb_img, (pan_img.shape))

        sharp_img = algorithm(pan_img=pan_img, rgb_img=rgb_img)
        score = get_pansharpening_scores(img, sharp_img)

        all_scores.append(score)

    all_scores = pd.concat(all_scores, axis=1, ignore_index=True)
    all_scores = all_scores.mean(axis=1)

    return all_scores


def evaluate_model_all_metrics(net, data_loader, sota=False):
    '''
        A function that evaluates a deep learning alhorithm on all scores.

        Args:
            net: The network that should be evaluated.
            data_loader: Data loader of the data that should be evaluated.
            sota (bool): Do images come from the SOTA dataset.
        
        Returns:
            (pd.DataFrame): A pandas DataFrame containing the scores.
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    net.eval()
    
    all_scores = []

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = net(x)
            x, y, outputs = x.to("cpu").numpy(), y.to("cpu").numpy(), outputs.to("cpu").numpy()

            for image, truth in zip(outputs, y):
                image = image.transpose((1, 2, 0))
                truth = truth.transpose((1, 2, 0))
                
                if not sota:
                    score = get_pansharpening_scores(np.array(image) * 4095, np.array(truth) * 4095)
                else:
                    score = get_pansharpening_scores(np.array(image) * 255, np.array(truth) * 255)

                all_scores.append(score)
            
    all_scores = pd.concat(all_scores, axis=1, ignore_index=True)
    all_scores = all_scores.mean(axis=1)

    return all_scores