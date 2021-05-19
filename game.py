# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""


import cv2
import tqdm
import consts
import numpy as np
from pathlib import Path
from player import Player
import matplotlib.pyplot as plt
from typing import Optional, List
from utils import (select_k,
                   is_image_file,
                   assign_dealer,
                   criterion_card_like,
                   order_anti_clockwise,
                   keep_contour_with_min_area,
                   keep_and_order_by_criterion)
from detection.segmentation import (find_contours,
                                    match_pattern,
                                    process_threshold)


class Game:
    """
    Enfold the logic around a game
    Use the segmentation, utils and player logic to process each step of the game
    """
    def __init__(self, path):
        self.path = path if isinstance(path, Path) else Path(path)
        self.P1 = None
        self.P2 = None
        self.P3 = None
        self.P4 = None
        self.players = [None] * 4
        self._paths = [p for p in self.path.iterdir() if is_image_file(p)]
        self.images = []
        self._roi_table = [None] * len(self._paths)
        self._current_step = -1
        self._processed_steps = []
        self.max_step = len(self._paths)
        self.is_done = False

    def load_images(self, verbose: Optional[bool] = None) -> List[np.ndarray]:
        """
        Load the images whose paths are in self._paths (created at initialization)
        :return: The list of loaded images
        """
        verbose = True if verbose is None else verbose
        self.images = []
        assert len(self._paths) > 0, f"{self.path.absolute()} contains no valid image file"
        print("Loading files") if verbose else None
        paths = tqdm.tqdm(self._paths) if verbose else self._paths
        errors = []
        for p in paths:
            try:
                img = cv2.imread(str(p.absolute()), cv2.IMREAD_UNCHANGED)
                self.images.append(img)

            except Exception:
                errors.append(str(p.absolute()))
                pass
        print(f"{len(self.images)} files loaded from {self.path.absolute()}") if verbose else None
        print(f"{len(errors)} error" + f"s while attempting to load these files : {errors}" * (
                len(errors) > 0)) if verbose else None
        return self.images

    def process_step(self, k: int,
                     show_step: Optional[bool] = None):
        """
        Process step k of the game and return the ROI of the players (dimension taken in consts.py)
        and the the overlay of players' name and contours
        :param show_step: Whether to show the process step step
        :param k: Step to process
        """
        assert -1 < k < len(self.images), f"the step must be between {-1} and {len(self.images)} but you provided {k}"
        src = self.images[k]
        show_step = False if show_step is None else show_step
        bw = process_threshold(src, True, min_val=consts.MIN_THRESHOLD_VAL)
        contours = find_contours(bw)
        contours = keep_and_order_by_criterion(contours, criterion_card_like)
        contours = keep_contour_with_min_area(contours, consts.MIN_CARD_AREA)
        contours, success = select_k(contours, consts.CARD_MIN_DISTANCE)
        anti_clockwise_ordered_contours = order_anti_clockwise(contours)
        a_c_o = anti_clockwise_ordered_contours
        self.P1 = Player(a_c_o[0], consts.Ordinal.SOUTH, "Player 1")
        self.P2 = Player(a_c_o[1], consts.Ordinal.EAST, "Player 2")
        self.P3 = Player(a_c_o[2], consts.Ordinal.NORTH, "Player 3")
        self.P4 = Player(a_c_o[3], consts.Ordinal.WEST, "Player 4")
        self.players = [self.P1, self.P2, self.P3, self.P4]
        dealer = cv2.imread(consts.DEALER_PATTERN_PATH, cv2.IMREAD_UNCHANGED)
        box, center, _, _ = match_pattern(src, dealer)
        assign_dealer(self.players, dealer_center=center)
        roi_p1 = self.P1.roi_centered(src, consts.H, consts.W)
        roi_p2 = self.P2.roi_centered(src, consts.H, consts.W)
        roi_p3 = self.P3.roi_centered(src, consts.H, consts.W)
        roi_p4 = self.P4.roi_centered(src, consts.H, consts.W)
        cv2.drawContours(src, [box], -1, consts.BOUNDING_BOX_COLOR, consts.BOUNDING_BOX_THICKNESS)
        self.P1.write_text(src, "P1")
        self.P2.write_text(src, "P2")
        self.P3.write_text(src, "P3")
        self.P4.write_text(src, "P4")
        roi = [roi_p1, roi_p2, roi_p3, roi_p4]
        self._roi_table[k] = roi
        if show_step:
            plt.title(f"Game step {k}")
            plt.imshow(src)
            plt.show()
        return roi, src

#fixme dev suspendu (no use)
    def initialize_game(self):
        """

        :return:
        """
        return None

    def next_step(self):
        """
        Process the next step of the game and update the current_step and game status
        :return: either None, None if game is ended or ROI, dest
        """
        if self.is_done:
            return None, None
        if self._current_step < self.max_step:
            ret = self.process_step(self._current_step + 1)
            self._current_step += 1
            self.is_done = self._current_step + 1 == self.max_step
            return ret
        else:
            self.is_done = True
            return None, None