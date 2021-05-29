# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
import cv2
from enum import Enum
from pathlib import Path


class Container:
    pass


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


detector_const = Container()
detector_const.card_mapping = {0: '0C', 1: '0D', 2: '0H', 3: '0S', 4: '1C', 5: '1D', 6: '1H', 7: '1S', 8: '2C', 9: '2D',
                               10: '2H', 11: '2S', 12: '3C',
                               13: '3D', 14: '3H', 15: '3S', 16: '4C', 17: '4D', 18: '4H', 19: '4S', 20: '5C', 21: '5D',
                               22: '5H', 23: '5S', 24: '6C', 25: '6D',
                               26: '6H', 27: '6S', 28: '7C', 29: '7D', 30: '7H', 31: '7S', 32: '8C', 33: '8D', 34: '8H',
                               35: '8S', 36: '9C', 37: '9D', 38: '9H',
                               39: '9S', 40: 'JC', 41: 'JD', 42: 'JH', 43: 'JS', 44: 'KC', 45: 'KD', 46: 'KH', 47: 'KS',
                               48: 'QC', 49: 'QD', 50: 'QH', 51: 'QS'}
detector_const.family_mapping = {'D': 0, 'H': 1, 'C': 2, 'S': 3}
detector_const.rank_mapping = {'K': 3, 'Q': 2, 'J': 1, '9': 0, '8': 0, '7': 0, '6': 0, '5': 0, '4': 0, '3': 0, '2': 0,
                               '1': 0, '0': 0}
detector_const.rank_detector_path = './weights/rank_detector'
detector_const.mnist_detector_path = './weights/mnist_detector'
detector_const.family_detector_path = './weights/family_detector'

RANKS = "0123456789JQK"
RANKS_NUMBER = {RANKS[k]: k for k in range(len(RANKS))}
H = 860
W = 600
# H = 768
# W = 512
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
DEFAULT_SAVE_DIR = Path() / "results"
