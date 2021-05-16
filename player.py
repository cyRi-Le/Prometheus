# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
from typing import Iterable, Optional, List, Tuple

import cv2
import numpy as np
from enum import Enum
import consts
from consts import Position, Ordinal




class Player:
    def __init__(self, contour: List,
                 ordinal: Ordinal,
                 name: Optional[str] = "P"):
        self.contour = contour
        self.min_rect = cv2.minAreaRect(contour)
        self.box = cv2.boxPoints(self.min_rect)
        self.angle = self.min_rect[2]
        self.center = self.min_rect[0]
        self.centerx, self.centery = self.center
        self.width = self.min_rect[1][0]
        self.height = self.min_rect[1][1]
        self.name = name
        self.ordinal = ordinal
        self.is_dealer = False

    def write_text(self,
                   img: np.ndarray,
                   text: Optional[str] = None,
                   **kwargs):
        """

        :param img:
        :param text:
        :param kwargs:
        :return:
        """
        text = text if text is not None else self.name
        # font
        font = consts.FONT
        # org
        if self.ordinal is Ordinal.EAST or self.ordinal is Ordinal.WEST:
            org = (int(self.centerx), int(self.centery) + int(self.height / 2))
        else:
            org = (int(self.centerx) + int(self.width / 2), int(self.centery))

        # fontScale
        fontScale = consts.FONT_SCALE
        # Red color in BGR
        color = consts.FONT_COLOR if not self.is_dealer else consts.DEALER_FONT_COLOR
        # Line thickness of 2 px
        thickness = consts.FONT_THICKNESS
        # Using cv2.putText() method
        image = cv2.putText(img, text, org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)
        return image

    def greetings(self):
        """

        """
        if self.ordinal is not Ordinal.NONE:
            print(f"I am {self.name} siding {self.ordinal.name}" + " I am the dealer" * self.is_dealer)
        else:
            print(f"I am {self.name}." + " I am the dealer" * self.is_dealer)

    def distance_to(self, contour: List,
                    target_center: Optional[Tuple] = None) -> float:
        """
        :param target_center:
        :param contour:
        :return:
        """
        target_center = target_center if target_center is not None else cv2.minAreaRect(contour)[0]
        return ((self.centerx - target_center[0]) ** 2 + (self.centery - target_center[1]) ** 2) ** 0.5

    # fixme
    def standard_box(self, h, w):
        """
        The common misconception of the "box" values is that the first sub-list of the "box" ndarray
        is always the bottom-left point of the rectangle.
        :param w:
        :param h:
        :return:
        """
        angle = self.angle
        if self.width > self.height:
            w, h = sorted([w, h], reverse=True)
        else:
            w, h = sorted([w, h])
        # angle = self.angle if self.width > self.height else 90 - self.angle
        min_rect = (self.center, (w, h), angle)
        return cv2.boxPoints(min_rect)

    # @private
    def _get_box_points_order(self) -> List[Position]:
        """

        :param box_points:
        :return:
        """
        ordered = [None] * 4
        box_points = self.box
        left1, left2, right1, right2 = np.argsort([point[0] for point in box_points])
        if box_points[left1][1] <= box_points[left2][1]:
            ordered[left1] = Position.TOP_LEFT
            ordered[left2] = Position.BOTTOM_LEFT
        else:
            ordered[left2] = Position.TOP_LEFT
            ordered[left1] = Position.BOTTOM_LEFT

        if box_points[right1][1] <= box_points[right2][1]:
            ordered[right1] = Position.TOP_RIGHT
            ordered[right2] = Position.BOTTOM_RIGHT
        else:
            ordered[right2] = Position.TOP_RIGHT
            ordered[right1] = Position.BOTTOM_RIGHT
        return ordered

    # @private
    def _new_box_points(self, dimx, dimy):
        """

        :param dimx:
        :param dimy:
        :param box_points:
        :param ordered:
        :return:
        """
        ordered = self._get_box_points_order()
        return [np.array(t.value) * (dimx - 1, dimy - 1) for t in ordered]

    def roi_centered(self, img, h, w):
        """

        :param img:
        :param w:
        :param h:
        :return:
        """
        std_box = np.int0(self.standard_box(w, h))
        src_box = self.standard_box(w, h).astype('float32')
        h_, w_ = h, w
        if self.width > self.height:
            w, h = sorted([w, h], reverse=True)
        else:
            w, h = sorted([w, h])

        dimx = max(w, h) if (self.ordinal is Ordinal.WEST or self.ordinal is Ordinal.EAST) else min(w, h)
        dimy = min(w, h) if (self.ordinal is Ordinal.WEST or self.ordinal is Ordinal.EAST) else max(w, h)

        # fixme on peut modifier pour que le ROI respecte le sens de lecture de la carte
        # fixme donner le meme sens de lecture sinon on peut deformer la carte
        dst_box = np.array(self._new_box_points(dimx, dimy), dtype="float32")
        rotate = (self.ordinal is Ordinal.EAST or self.ordinal is Ordinal.WEST)
        # the perspective transformation matrix
        M = cv2.getPerspectiveTransform(src_box, dst_box)
        size = (w_, h_) if not rotate else (h_, w_)
        warped = cv2.warpPerspective(img.copy(), M, size)
        cv2.drawContours(img, [std_box], -1, consts.BOUNDING_BOX_COLOR, consts.BOUNDING_BOX_THICKNESS)
        return warped if not rotate else cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
