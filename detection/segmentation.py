# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
from typing import Iterable, Optional, Callable, List, Tuple
import numpy as np
import cv2


def process_threshold(img: np.ndarray,
                      to_gray_level: Optional[bool] = False,
                      method: Optional = None,
                      min_val: Optional[int] = 128,
                      max_val: Optional[int] = 255):
    """

    :param to_gray_level:
    :param img:
    :param method:
    :param min_val:
    :param max_val:
    :return:
    """
    method = method if method is not None else cv2.THRESH_OTSU + cv2.THRESH_BINARY
    img = img if not to_gray_level else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dest = cv2.threshold(img, min_val, max_val, method)
    return dest


def find_threshold(img: np.ndarray):
    """

    :param img:
    :return:
    """
    return None


def find_contours(img: np.ndarray,
                  process: Optional[Callable] = None) -> List:
    """

    :param img:
    :param process:
    :return:
    """
    contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #contours = np.array(contours, dtype=object)
    return contours if process is None else process(contours)


def match_pattern(img: np.ndarray,
                  pattern: np.ndarray,
                  match_method: Optional[int] = None) -> Tuple:
    """

    :param img:
    :param pattern:
    :param match_method:
    :return:
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


