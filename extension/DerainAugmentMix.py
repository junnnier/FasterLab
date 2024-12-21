import numpy as np
from PIL import Image


def int_parameter(level, maxval):
    """
    Helper function to scale `val` between 0 and maxval .
    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to level/PARAMETER_MAX.
    Returns:
        An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """
    Helper function to scale `val` between 0 and maxval.
    Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to level/PARAMETER_MAX.
    Returns:
        A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform((pil_img.width, pil_img.height), Image.AFFINE, (1, level, 0, 0, 1, 0), resample=Image.BILINEAR)


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
      level = -level
    return pil_img.transform((pil_img.width, pil_img.height), Image.AFFINE, (1, 0, 0, level, 1, 0), resample=Image.BILINEAR)


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.width / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((pil_img.width, pil_img.height), Image.AFFINE, (1, 0, level, 0, 1, 0), resample=Image.BILINEAR)


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), pil_img.height / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform((pil_img.width, pil_img.height), Image.AFFINE, (1, 0, 0, 0, 1, level), resample=Image.BILINEAR)


def zoom_x(pil_img, level):
    level = float_parameter(sample_level(level), 6.0)
    rate = 1.0 / level
    if np.random.random() > 0.5:
        bias = pil_img.width * (1 - rate)
    else:
        bias = 0
    return pil_img.transform((pil_img.width, pil_img.height), Image.AFFINE, (rate, 0, bias, 0, 1, 0), resample=Image.BILINEAR)


def zoom_y(pil_img, level):
    level = float_parameter(sample_level(level), 6.0)
    rate = 1.0 / level
    if np.random.random() > 0.5:
        bias = pil_img.height * (1 - rate)
    else:
        bias = 0
    return pil_img.transform((pil_img.width, pil_img.height), Image.AFFINE, (1, 0, 0, 0, rate, bias), resample=Image.BILINEAR)


augmentations = [
    rotate, shear_x, shear_y,
    translate_x, translate_y, zoom_x, zoom_y
]


def normalize(image):
    """Normalize input image channel-wise to zero mean and unit variance."""
    '''
    image = image.transpose(2, 0, 1)  # Switch to channel-first
    mean, std = np.array(MEAN), np.array(STD)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)
    '''
    return image


def apply_op(image, op, severity):
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    pil_img = Image.fromarray(image)  # Convert to PIL.Image
    pil_img = op(pil_img, severity)
    return np.asarray(pil_img) / 255.


def derain_augment_mix(image, severity=3, width=3, depth=-1, alpha=1.):
    """
    Args:
        image: Raw input image as float32 np.ndarray of shape (h, w, c)
        severity: Severity of underlying augmentation operators (between 1 to 10).
        width: Width of augmentation chain
        depth: Depth of augmentation chain. -1 enables stochastic depth uniformly from [1, 3]
        alpha: Probability coefficient for Beta and Dirichlet distributions.
    Returns:
        Augmented and mixed image.
    """
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    mix = np.zeros_like(image)
    for i in range(width):
        image_aug = image.copy()
        depth = depth if depth > 0 else np.random.randint(2, 4)
        for _ in range(depth):
            op = np.random.choice(augmentations)
            image_aug = apply_op(image_aug, op, severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * normalize(image_aug)

    max_ws = max(ws)
    rate = 1.0 / max_ws
    mixed = max((1 - m), 0.7) * normalize(image) + max(m, rate * 0.5) * mix
    return mixed