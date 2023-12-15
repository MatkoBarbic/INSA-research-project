import numpy as np
import cv2
from utils import downsample, apply_psf, process_pansharp_img

def brovy(pan_img: np.array, rgb_img: np.array, weights: list =[0.33, 0.33, 0.33]):
    '''
        A function that performs brovy pansharpening.

        Args:
            pan_img (np.array[int]): High resolution panchromatic (grayscale) image.
            rgb_img (np.array[int]): Low resolution multispectral (RGB) image.
            weights (list[int]): Weights for the red, green, and blue channels respectively.

        Returns:
            (np.array[int]): Pansharpened image.
    '''
    
    dnf = pan_img / np.sum(rgb_img * weights, axis=2)
    dnf = dnf[:, :, np.newaxis]

    sharp_img = rgb_img * dnf
    
    return sharp_img


def ihs(pan_img: np.array, rgb_img: np.array):
    '''
        A function that performs IHS pansharpening.

        Args:
            pan_img (np.array[int]): High resolution panchromatic (grayscale) image.
            rgb_img (np.array[int]): Low resolution multispectral (RGB) image.

        Returns:
            (np.array[int]): Pansharpened image.
    '''
    hls_image = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HLS)
    hls_image[:, :, 1] = pan_img
    sharp_img = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)
    
    return sharp_img


def regularize_images(img1, img2):
    mean_1 = img1.mean()
    mean_2 = img2.mean()
    std_1 = img1.std()
    std_2 = img2.std()

    img1 = ((img1 - mean_1) / std_1) * std_2 + mean_2
    return img1


def projection(v: np.array, u: np.array):
    '''
        A function that calculates the projection of a vector v on another vector u.

        Args:
            v (np.array[float]): Vector to be projected.
            u (np.array[float]): Vector on which the vector v will be projected.

        Returns:
            (np.array[float]): Projected vector.
    '''
    return ((v @ u) / (u @ u)) * u


def gram_schmidt_orth(a: np.array):
    '''
        A function that performs brovy pansharpening.

        Args:
            a (np.array[float]): Multidimensional array to be orthogonalized.

        Returns:
            (np.array[float]): Orthogonalized array.
    '''
    rows, columns = a.shape

    orthogonal_basis = []
    
    for c in range(columns):
        new_base = a[:, c]
        for base in orthogonal_basis:
            new_base -= projection(v=new_base, u=base)
        orthogonal_basis.append(new_base)

    orthogonal_basis = np.array(orthogonal_basis)
    norm_orth_basis = orthogonal_basis / np.linalg.norm(orthogonal_basis, axis=1, keepdims=True)
    
    return norm_orth_basis.T


def gram_schmidt_pan(pan_img: np.array, rgb_img: np.array, weights: list =[0.33, 0.33, 0.33]):
    '''
        A function that performs Gram-Schmidt pansharpening.

        This algorithm takes in vectors (for example, three vectors in 3D space) that are not orthogonal, and then rotates them so that they are orthogonal afterward. In the case of images, each band (panchromatic, red, green, blue, and infrared) corresponds to one high-dimensional vector (number of dimensions equals number of pixels)
        
        Steps:
            1. Create a low-resolution pan band by computing a weighted average of the multi-spectral bands

            2. These bands are decorrelated using the Gram-Schmidt orthogonalization algorithm, treating each band as one multidimensional vector with simulated low-resolution pan band is used as the first vector
            
            3. Low-resolution pan band is then replaced by the high-resolution pan band
            
            4. All bands are back-transformed in high resolution

        Some suggested weights for common sensors are as follows (red, green, blue, and infrared, respectively):
            GeoEye—0.6, 0.85, 0.75, 0.3
            IKONOS—0.85, 0.65, 0.35, 0.9
            QuickBird—0.85, 0.7, 0.35, 1.0
            WorldView-2—0.95, 0.7, 0.5, 1.0

        Args:
            pan_img (np.array[int]): High resolution panchromatic (grayscale) image.
            rgb_img (np.array[int]): Low resolution multispectral (RGB) image.
            weights (list[int]): Weights for the red, green, and blue channels respectively.

        Returns:
            (np.array[int]): Pansharpened image.
    '''
    original_basis = np.identity(3)
    original_basis[:, 0] = np.array(weights)

    orth_basis = gram_schmidt_orth(a=original_basis)

    # # Uncomment to check if the basis is orthogonal
    # # product = np.dot(orth_basis, orth_basis.T)
    # # identity_matrix = np.eye(orth_basis.shape[0])

    # # print(np.allclose(product, identity_matrix))

    orth_bands = rgb_img @ orth_basis
    pan_img = regularize_images(pan_img, orth_bands[:, :, 0])
    orth_bands[:, :, 0] = pan_img

    reverse_coefs = np.linalg.inv(orth_basis)
    
    sharp_img = orth_bands @ reverse_coefs

    return sharp_img


def pansharp_pipeline(path_to_image: str, psf: np.array, downsampling_factor: int, algorithm: callable):
    '''
        A function that creates a pansharpening pipeline.
        The pipeline consists of:
            1. Creating panchromatic (grayscale) version of the image.
            2. Downsampling the panchromatic and the rgb images.
            3. Performing a pansharpening algorithm on the images.

        Args:
            path_to_image (str): Path to original pansharpened data.
            psf (np.array[int]): PSF array to be applied to the original image.
            downsampling_factor (int): Factor by which the images is to be downsampled.
            algorithm (callable): A pansharpening algorithm which will be performed on the images.

        Returns:
            pan_img (np.array[int]): Downscaled panchromatic image.
            rgb_img (np.array[int]): Downscaled RGB image
            sharp_img (np.array[int]): Pansharpened image.
    '''
    pan_img, rgb_img, original_img = process_pansharp_img(path_to_image=path_to_image, psf=psf)

    sharp_img = algorithm(pan_img=pan_img, rgb_img=rgb_img)

    return pan_img, rgb_img, sharp_img