# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
import os
import cv2
import consts
import numpy as np
from pathlib import Path
from player import Player
from typing import Optional, List, Tuple, Callable


def assign_dealer(players: List[Player],
                  dealer_contour: Optional[List] = None,
                  dealer_center: Optional[Tuple] = None):
    """
    Assign dealer to one of the given players based on minimum distance
    from player's contour to the dealer mark
    Once the dealer is assigned, the player prints a greeting message :)
    :param dealer_center: Optional coordinates of the dealer mark center
    :param players: List of players
    :param dealer_contour: contour of the dealer mark given by cv2.findContours()
    """
    dist = [player.distance_to(dealer_contour, dealer_center) for player in players]
    dealer = np.argmin(dist)
    players[dealer].is_dealer = True
    players[dealer].greetings()


def bounding_box(contour: List):
    """
    Return the box points of the minimum enclosing rectangle
    designed to be used in cv2.drawContours()
    :param contour: contour given by cv2.findContours()
    :return: box points of the minimum enclosing rectangle
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box


def circle_box(contour: List):
    """
    Return the center and the radius of the minimum enclosing circle
    designed to be used in cv2.circle()
    :param contour: contour given by cv2.findContours()
    :return: center and radius of the minimum enclosing circle
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

    :param min_rect: Optional rectangle given by cv2.minAreaContour()
    :param min_val_2: Minimum value of feature 2
    :param min_val_1: Minimum value of feature 1
    :param max_val_1: Maximum value of feature 1
    :param contour: contour given by cv2.findContours()
    :return bool & float: Whether the contour is valid or not and the values of feature
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
    Estimate how circle-like is the given contour
    :param min_val: Minimum value of the feature
    :param max_val: Maximum value of the feature
    :param contour: contour given by cv2.findContours()
    :return bool & float: Whether the contour is valid or not and the value of feature
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
    Compute the area of the contour w.r.t to its validity given by criterion
    :param criterion: Function which return a tuple of boolean (first) and a value (second)
    :param contour: contour given by cv2.findContours()
    :return area: either 0. (contour is not valid for criterion) or cv2.contourArea()
    if the contour is valid for the given criterion
    """
    if criterion(contour, **kwargs)[0]:
        return cv2.contourArea(contour)
    return 0.


def keep_contour_with_min_area(contours: List[List],
                               min_area: float) -> List[List]:
    """
    Keep contours whose area is at least min_area
    :param contours: List of contours given by cv2.findContours()
    :param min_area: Minimum area
    """
    return [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]


def order_anti_clockwise(contours: List[List],
                         min_rects: Optional[List[Tuple]] = None) -> Tuple[list, list, list, list]:
    """
    Rank a list of 4 contours in anti-clockwise order
    :param min_rects: Optional List of rectangles given by cv2.minAreaContour()
    :param contours: List of contours given by cv2.findContours()
    :return contours: In anti-clockwise ordering
    """
    assert len(contours) == 4, f"order_anti_clockwise takes a list of 4 contours but {len(contours)} where given"
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
    Select the k first contours mutually distant from at least min_distance
    :param min_rects: Optional List of rectangles given by cv2.minAreaContour()
    :param contours: List of contours given by cv2.findContours()
    :param min_distance: Minimum mutual distance
    :param k: number of element to return
    :return selected: k first contours mutually distance from at least min_distance
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
    Build symmetric matrix where element i,j represents the distance from
    contour i center  to contour j center
    :param min_rects: Optional List of rectangles given by cv2.minAreaContour()
    :param contours:  List of contours given by cv2.findContours()
    :return map + map.T: Symmetric matrix of distance between contours
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
    Given a criterion the function the elements which
    satisfies the criterion in ascending order of criterion values

    :param contours: List of contours given by cv2.findContours()
    :param criterion: Function which return a tuple of boolean (first) and a value (second)
    :param reverse: Ascending (False) or descending (True) order
    :param kwargs: Any additional parameters to give to the function criterion
    :return contours: Valid and ordered contours by criterion
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


def is_image_file(path) -> bool:
    """
    Check if the to path correspond to a valid image file
    :param path: PosixPath or str
    :return: boolean True if path corresponds to an image file, False otherwise
    """
    path = path if isinstance(path, Path) else Path(path)
    _, ext = os.path.splitext(path.absolute())
    return path.is_file() and ext.lower() in consts.IMAGE_FILE_EXT
