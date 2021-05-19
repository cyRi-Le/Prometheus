# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
import cv2
import consts
import numpy as np
from consts import Position, Ordinal
from typing import Optional, List, Tuple


class Player:
    """
    Enfold the logic around a player
    This class is designed to be use extensively in the Game class
    """
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
                   text: Optional[str] = None):
        """
        Write a text close to the player location. The text is rendered on the given image
        :param img: Image to write on
        :param text: Text to write
        :return: Image with written text
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
        # fontColor
        color = consts.FONT_COLOR if not self.is_dealer else consts.DEALER_FONT_COLOR
        # Line thickness
        thickness = consts.FONT_THICKNESS
        image = cv2.putText(img, text, org, font, fontScale,
                            color, thickness, cv2.LINE_AA, False)
        return image

    def greetings(self):
        """
        Print a greeting message once
        If the player is the dealer it will tell it (You should trust him :)
        """
        if self.ordinal is not Ordinal.NONE:
            print(f"I am {self.name} siding {self.ordinal.name}" + " I am the dealer" * self.is_dealer)
        else:
            print(f"I am {self.name}." + " I am the dealer" * self.is_dealer)

    def distance_to(self, contour: List,
                    target_center: Optional[Tuple] = None) -> float:
        """
        Compute the distance from the player's contour center to a target contour center
        :param target_center: Optional center of the target contour
        :param contour: Contour given by cv2.findContours()
        :return float : Euclidean distance from the player to the target contour
        """
        target_center = target_center if target_center is not None else cv2.minAreaRect(contour)[0]
        return ((self.centerx - target_center[0]) ** 2 + (self.centery - target_center[1]) ** 2) ** 0.5

    # fixme
    def standard_box(self, h, w):
        """
        Return boxPoints corresponding to the rectangle of dimensions h, w
        concentric with the players cv2.minAreaContour() and same orientation

        WARNING: The box points returned by cv2.boxPoints(min_rect) are
        not always in the same order i.e (top-left -> bottom-left etc.) Their order depends on
        what is called width / height and the value of angle in the tuple given by cv2.minAreaContour()
        Two rectangle can have the same visual orientation and dimensions but their cv2.boxPoints() could be
        in very different orientations. So the boxPoints returned by this function are not assumed to be
        in the orientation of the player's boxPoints. That's why we need _get_box_points_order() to further
        rearrange the points before doing a Perspective Transform
        :param w: Width (automatically adapted to type of player)
        :param h: Height (automatically adapted to type of player)
        :return: Enlarged rectangle's boxPoints
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
        Return the order of the player's box points
        :return: A list of position where the i-th element is
        the Position of the i-th point the player's box points
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
        Return a new box points and most importantly the points are in the
        same order of the box points as player's box points
        :param dimx: Dimension along x axis
        :param dimy: Dimension along y axis
        :return: Box points oriented in the same order
        """
        ordered = self._get_box_points_order()
        return [np.array(t.value) * (dimx - 1, dimy - 1) for t in ordered]

    def roi_centered(self, img, h, w):
        """
        Return and draw the rectangle of the ROI of dimensions h,w of the player on img
        The ROI is oriented is returned the way the player sees it from its seat
        :param img: Given img to draw ROI rectangle on and extract cropped ROI
        :param w: Width (automatically adapted to type of player)
        :param h: Height (automatically adapted to type of player)
        :return: The ROI of the player always in same dimensions no matter the player
        """
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
        cv2.drawContours(img, [np.int0(src_box)], -1, consts.BOUNDING_BOX_COLOR, consts.BOUNDING_BOX_THICKNESS)
        return warped if not rotate else cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
