"""
image_ops.py  —  Task 2: Section A Image Processing
----------------------------------------------------
OpenCV utilities for all Section A operations:
  - Image blending (three alpha ratios)
  - Histogram equalisation
  - Brightness adjustment
  - Horizontal / vertical flipping
  - 90-degree clockwise rotation

All functions accept and return BGR numpy arrays (as loaded by cv2.imread).
Convert to RGB before passing to matplotlib: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
"""

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# A.1  Image Blending
# ---------------------------------------------------------------------------

def resize_to_match(img_a: np.ndarray, img_b: np.ndarray):
    """Resize img_b to match the height and width of img_a."""
    h, w = img_a.shape[:2]
    img_b_resized = cv2.resize(img_b, (w, h), interpolation=cv2.INTER_LINEAR)
    return img_a, img_b_resized


def blend(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    """
    Blend two images using a weighted sum.

    Formula: output(x,y) = alpha * img_a(x,y) + (1 - alpha) * img_b(x,y)

    Args:
        img_a:  First image (BGR numpy array).
        img_b:  Second image (BGR numpy array). Resized to match img_a if needed.
        alpha:  Weight for img_a in range [0.0, 1.0].
                (1 - alpha) is the weight for img_b.

    Returns:
        Blended BGR image as uint8 numpy array.
    """
    img_a, img_b = resize_to_match(img_a, img_b)
    return cv2.addWeighted(img_a, alpha, img_b, 1.0 - alpha, gamma=0)


# ---------------------------------------------------------------------------
# A.2  Histogram Equalisation
# ---------------------------------------------------------------------------

def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """
    Apply histogram equalisation to a colour image via the grayscale channel.

    Steps:
      1. Convert BGR -> GRAY
      2. Apply cv2.equalizeHist (CDF-based intensity remapping)
      3. Convert GRAY -> BGR so the result has the same shape as the input

    The equalised grayscale image is returned as a 3-channel BGR array to
    allow side-by-side display alongside the original colour image.

    Args:
        img: BGR numpy array (colour or grayscale).

    Returns:
        Equalised image as 3-channel BGR uint8 array.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    eq   = cv2.equalizeHist(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# A.3  Image Processing Techniques
# ---------------------------------------------------------------------------

def adjust_brightness(img: np.ndarray, beta: int) -> np.ndarray:
    """
    Shift the brightness of every pixel by a constant offset.

    Formula: output(x,y) = clip(img(x,y) + beta, 0, 255)

    Uses cv2.convertScaleAbs with alpha=1.0 (no contrast change) and
    beta as the brightness offset. Pixel values are clipped to [0, 255].

    Args:
        img:  BGR numpy array.
        beta: Integer offset. Positive = brighter, negative = darker.
              Typical range: -100 to +100.

    Returns:
        Brightness-adjusted BGR image.
    """
    return cv2.convertScaleAbs(img, alpha=1.0, beta=beta)


def flip_image(img: np.ndarray, direction: str) -> np.ndarray:
    """
    Flip an image horizontally or vertically.

    Args:
        img:       BGR numpy array.
        direction: 'horizontal' — mirror left-right  (cv2.flip flipCode=1)
                   'vertical'   — mirror top-bottom  (cv2.flip flipCode=0)

    Returns:
        Flipped BGR image.

    Raises:
        ValueError: If direction is not 'horizontal' or 'vertical'.
    """
    flip_codes = {'horizontal': 1, 'vertical': 0}
    if direction not in flip_codes:
        raise ValueError(f"direction must be 'horizontal' or 'vertical', got '{direction}'")
    return cv2.flip(img, flip_codes[direction])


def rotate_90cw(img: np.ndarray) -> np.ndarray:
    """
    Rotate an image 90 degrees clockwise using an affine transformation.

    Steps:
      1. Compute rotation matrix: centre = (w/2, h/2), angle = -90, scale = 1
      2. Adjust the translation component so the rotated image fits the new canvas
      3. Apply cv2.warpAffine with swapped output dimensions (new_w=h, new_h=w)

    Using a full affine warp (rather than cv2.rotate) ensures the bounding box
    is correctly handled for non-square images.

    Args:
        img: BGR numpy array of shape (H, W, C).

    Returns:
        Rotated BGR image of shape (W, H, C).
    """
    h, w = img.shape[:2]
    centre = (w / 2, h / 2)

    # Rotation matrix for -90 degrees (clockwise)
    M = cv2.getRotationMatrix2D(centre, angle=-90, scale=1.0)

    # Adjust translation so the output fits within (new_w=h, new_h=w)
    M[0, 2] += (h - w) / 2
    M[1, 2] += (w - h) / 2

    # new canvas size: width becomes old height, height becomes old width
    return cv2.warpAffine(img, M, (h, w))
