import numpy as np
import pywt
import cv2

def w2d(img, mode='haar', level=1):
    """
    Perform wavelet transform on an image and return the detail coefficients reconstruction.

    Parameters:
        img (np.ndarray): Input image in RGB format.
        mode (str): Type of wavelet to use (default is 'haar').
        level (int): Decomposition level (default is 1).

    Returns:
        np.ndarray: Reconstructed image from detail coefficients (uint8 grayscale).
    """
    # Convert to grayscale
    imArray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Convert to float and normalize
    imArray = np.float32(imArray)
    imArray /= 255

    # Compute wavelet coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    # Zero out approximation coefficients to focus on details
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # Reconstruct image from detail coefficients
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H
