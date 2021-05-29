# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""

import cv2
import numpy as np
from typing import Optional, Callable, List, Tuple


def process_threshold(img: np.ndarray,
                      to_gray_level: Optional[bool] = False,
                      method: Optional = None,
                      min_val: Optional[int] = 128,
                      max_val: Optional[int] = 255):
    """
    Apply a threshold to the BGR or Grayscale image
    :param to_gray_level: boolean If True the image is assumed to
    be in BGR and it is converted to Grayscale
    :param img: Image to process thresholding on
    :param method: thresholding technique to use (must be implemented in OpenCV)
    :param min_val: Minimum pixel intensity
    :param max_val: Maximum pixel intensity
    :return: process image
    """
    method = method if method is not None else cv2.THRESH_BINARY + cv2.THRESH_OTSU
    img = img if not to_gray_level else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dest = cv2.threshold(img, min_val, max_val, method)
    return dest


# fixme implementation suspendue (no use)
def find_threshold(img: np.ndarray):
    """

    :param img:
    :return:
    """
    return None


def find_contours(img: np.ndarray,
                  process: Optional[Callable] = None,
                  **kwargs) -> List:
    """
    Find the contours and pass them (if given) to a precessing function
    :param img: Image on which to find contours
    :param process: Processing function must take contours as first parameter and optional kwargs
    :return: contours or processed contours
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours if process is None else process(contours, **kwargs)


def match_pattern(img: np.ndarray,
                  pattern: np.ndarray,
                  match_method: Optional[int] = None) -> Tuple:
    """
    Look for a given pattern on the image using the given method (implemented in OpenCV)
    :param img: Image on which to look for contours
    :param pattern: Pattern to look for
    :param match_method: Method to use (one the methods implemented in OpenCV)
    :return: The most probable box points, location, cropping of the pattern on the image
    """
    match_method = match_method if match_method is not None else cv2.TM_CCORR_NORMED
    res = cv2.matchTemplate(img, pattern, match_method)
    cv2.normalize(res, res, 0, 1, cv2.NORM_MINMAX, -1)
    _, _, minLoc, maxLoc = cv2.minMaxLoc(res, None)
    dimy, dimx = pattern.shape[:2]
    if match_method == cv2.TM_SQDIFF or match_method == cv2.TM_SQDIFF_NORMED:
        matchLoc = minLoc
    else:
        matchLoc = maxLoc
    center = (matchLoc[0] + dimx / 2, maxLoc[1] + dimy / 2)
    ref_x, ref_y = maxLoc
    box_points = np.int0([maxLoc, (ref_x + dimx, ref_y), (ref_x + dimx, ref_y + dimy), (ref_x, ref_y + dimy)])
    return box_points, center, img[maxLoc[1]: matchLoc[1] + dimy, maxLoc[0]: matchLoc[0] + dimx, :], pattern

# TODO


# TODO adapter las angles au sens naturel
# TODO Adapter width et height a un sens conventionnel
