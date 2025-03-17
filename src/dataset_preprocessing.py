"""Dataset preprocessing module"""

import os
import cv2
import numpy as np

from skimage import exposure
from skimage.util import random_noise
from skimage.transform import rotate, resize
from skimage.filters import gaussian


def apply_gaussian_blur(image, sigma=1.0):
    """Gaussian Blur augmentation"""

    return gaussian(image, sigma=sigma, multichannel=True)


def apply_gaussian_noise(image, var=0.01):
    """Gaussian Noise augmentation"""

    return random_noise(image, mode="gaussian", var=var)


def adjust_contrast_brightness(image, contrast=1.0, brightness=0.0):
    """Contrast and brightness augmentation"""

    return exposure.adjust_gamma(image, gamma=contrast, gain=brightness)


def apply_rotation(image, angle=10):
    """Rotation augmentation"""

    return rotate(image, angle, resize=True)


def center_crop_resize(image, target_size=(512, 512)):
    """Center Crop Resize augmentation"""

    h, w = image.shape[:2]
    new_size = min(h, w)

    start_h = (h - new_size) // 2
    start_w = (w - new_size) // 2
    cropped = image[start_h : start_h + new_size, start_w : start_w + new_size]

    return resize(cropped, target_size)


def preprocess_image(image, target_size=(512, 512)):
    """Preprocess image"""

    # Apply random transformations
    image = apply_gaussian_blur(image, sigma=np.random.uniform(0, 0.2))
    image = apply_gaussian_noise(image, var=np.random.uniform(10, 50) / 255.0)
    image = adjust_contrast_brightness(
        image,
        contrast=np.random.uniform(0.8, 1.2),
        brightness=np.random.uniform(-0.2, 0.2),
    )
    image = apply_rotation(image, angle=np.random.uniform(-10, 10))
    image = center_crop_resize(image, target_size=target_size)

    # Normalize to [0, 1]
    image = np.clip(image, 0, 1)

    return image


def load_and_preprocess_image(image_path, target_size=(512, 512)):
    """Load and preprocess image"""

    # Load image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0

    # Preprocess image
    preprocessed_image = preprocess_image(image, target_size=target_size)

    return preprocessed_image
