# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
import cv2
from enum import Enum
# BEST VALUE 768 512
H = 860
W = 600

#H = 768
#W = 512
MIN_CIRCLE_AREA = 1e4
MIN_CARD_AREA = 1e5
MAX_CARD_AREA = 1e7
CARD_MIN_DISTANCE = 100
BOUNDING_BOX_COLOR = (0, 0, 159)
BOUNDING_BOX_THICKNESS = 10
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 5
FONT_COLOR = (0, 0, 153)
DEALER_FONT_COLOR = (0, 102, 51)
FONT_THICKNESS = 20
DEALER_PATTERN_PATH = "patterns/dealer.jpg"
IMAGE_FILE_EXT = [".jpg", ".jpeg", ".bmp", ".png"]
MIN_THRESHOLD_VAL = 150
MAX_FILE_NAME_LENGTH = 10
class Position(Enum):
    TOP_LEFT = (0, 0)
    TOP_RIGHT = (1, 0)
    BOTTOM_LEFT = (0, 1)
    BOTTOM_RIGHT = (1, 1)


class Ordinal(Enum):
    SOUTH = 0
    EAST = 1
    NORTH = 2
    WEST = 3
    NONE = -1
