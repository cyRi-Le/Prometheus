# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
from enum import Enum
from typing import Iterable, Optional, List, Tuple, Callable
import numpy as np

import consts
from player import Player
import cv2
import os
from pathlib import Path


def assign_dealer(players: List[Player],
                  dealer_contour: Optional[List] = None,
                  dealer_center: Optional[Tuple] = None):
    """

    :param dealer_center:
    :param players:
    :param dealer_contour:
    """
    dist = [player.distance_to(dealer_contour, dealer_center) for player in players]
    dealer = np.argmin(dist)
    players[dealer].is_dealer = True
    players[dealer].greetings()


def bounding_box(contour: List):
    """

    :param contour:
    :return:
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def circle_box(contour: List):
    """

    :param contour:
    :return:
    """
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)
    return center, radius


def criterion_card_like(contour: List,
                        min_rect: Optional[Tuple] = None,
                        min_val_1: Optional[float] = 1.3,
                        max_val_1: Optional[float] = 2.,
                        min_val_2: Optional[float] = 0.5):
    """

    :param min_rect:
    :param min_val_2:
    :param min_val_1:
    :param max_val_1:
    :param contour:
    :return:
    """
    min_rect = min_rect if min_rect is not None else cv2.minAreaRect(contour)
    dim1, dim2 = min_rect[1]
    f1 = max(dim1, dim2) / (1e-5 + min(dim1, dim2))

    # contour area
    s = cv2.contourArea(contour)
    # minimun rotated bounding box
    box = cv2.boxPoints(min_rect)
    s_box = cv2.contourArea(np.int0(box))
    f2 = s / (s_box + 1e-5)
    return min_val_1 < f1 < max_val_1 and f2 > min_val_2, f1, f2


def criterion_circle_like(contour: List,
                          min_val: Optional[float] = 0.8,
                          max_val: Optional[float] = 1.1):
    """

    :param min_val:
    :param max_val:
    :param contour:
    :return:
    """
    _, _, radius = cv2.minEnclosingCircle(contour)
    min_rect = cv2.minAreaRect(contour)
    w, h = min_rect[1]
    feature = ((w * h) ** 0.5) / (2 * radius + 1e-5)
    return min_val < feature < max_val, feature


def compute_area_criterion(contour: List,
                           criterion: Callable,
                           **kwargs):
    """

    :param criterion:
    :param contour:
    :return:
    """
    if criterion(contour, **kwargs)[0]:
        return cv2.contourArea(contour)
    return 0.


def keep_contour_with_min_area(contours: List[List],
                               min_area: float) -> List[List]:
    """

    :param contours:
    :param min_area:
    """
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]


def order_anti_clockwise(contours: List[List],
                         min_rects: Optional[List[Tuple]] = None) -> Tuple[list, list, list, list]:
    """

    :param min_rects:
    :param contours:
    :return:
    """
    min_rects = min_rects if min_rects is not None else [cv2.minAreaRect(contour) for contour in contours]
    SOUTH = np.argmax([min_rect[0][1] for min_rect in min_rects])
    NORTH = np.argmin([min_rect[0][1] for min_rect in min_rects])
    EAST = np.argmax([min_rect[0][0] for min_rect in min_rects])
    WEST = np.argmin([min_rect[0][0] for min_rect in min_rects])
    return contours[SOUTH], contours[EAST], contours[NORTH], contours[WEST]


def select_k(contours: List[List],
             min_distance: float,
             min_rects: Optional[List[Tuple]] = None,
             k: Optional[int] = 4) -> Tuple[List, bool]:
    """

    :param min_rects:
    :param contours:
    :param min_distance:
    :param k:
    :return:
    """
    distance_map = build_distance_map(contours, min_rects)
    n = len(distance_map)
    selected = [0]
    for i in range(n):
        acceptable = True
        for j in selected:
            if distance_map[i, j] < min_distance and i != j:
                acceptable = False
        if acceptable and i not in selected:
            selected.append(i)
        if len(selected) == k:
            break
    success = len(selected) == k
    return [contours[i] for i in selected], success


def build_distance_map(contours: List[List],
                       min_rects: Optional[List[Tuple]] = None) -> np.ndarray:
    """

    :param min_rects:
    :param contours:
    :return:
    """
    n = len(contours)
    min_rects = min_rects if min_rects is not None else [cv2.minAreaRect(contour) for contour in contours]
    map = np.zeros((n, n))
    for i in range(n):
        ref = min_rects[i]
        for j in range(n):
            if j < i:
                target = min_rects[j]
                dst = (ref[0][0] - target[0][0]) ** 2 + (ref[0][1] - target[0][1]) ** 2
                map[i, j] = dst ** 0.5
    return map + map.T


# fixme possibilite de retourner les features si ça peut servir
def keep_and_order_by_criterion(contours: List[List],
                                criterion: Callable,
                                reverse: Optional[bool] = True,
                                **kwargs) -> List[list]:
    """

    :param contours:
    :param criterion:
    :param reverse:
    :param kwargs:
    :return:
    """
    valid_contours_with_features = []
    for cnt in contours:
        valid_and_features = criterion(cnt, **kwargs)
        valid = valid_and_features[0]
        features = valid_and_features[1:]
        if valid:
            valid_contours_with_features.append((cnt, sum(features)))
    valid_contours_with_features = sorted(valid_contours_with_features, key=lambda cnt_and_feature: cnt_and_feature[1],
                                          reverse=reverse)
    contours = [cnt_with_feature[0] for cnt_with_feature in valid_contours_with_features]
    return contours


def is_image_file(path):
    """

    :param path:
    :return:
    """
    path = path if isinstance(path, Path) else Path(path)
    _, ext = os.path.splitext(path.absolute())
    return path.is_file() and ext in consts.IMAGE_FILE_EXT
