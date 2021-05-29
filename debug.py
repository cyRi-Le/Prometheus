# coding=utf-8
"""
PyCharm Editor
Author cyRi-Le
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import consts
from detection.segmentation import process_threshold, find_contours
from utils import keep_contour_with_min_max_area, select_k, keep_and_order_by_criterion, criterion_card_like, \
    bounding_box, compute_area

path = f"data/train_games/game5/8.jpg"
src = cv2.imread(path, cv2.IMREAD_UNCHANGED)


bw = process_threshold(src, True, min_val=consts.MIN_THRESHOLD_VAL)

contours = find_contours(bw)
contours = keep_and_order_by_criterion(contours, criterion_card_like)
for cnt in contours:
    print(cv2.contourArea(cnt)) if 50000 < cv2.contourArea(cnt) < 1e5 else None
contours = [cnt for cnt in contours if consts.MIN_CARD_AREA < compute_area(cnt)< consts.MAX_CARD_AREA]#keep_contour_with_min_max_area(contours, consts.MIN_CARD_AREA, consts.MAX_CARD_AREA)
#contours, success = select_k(contours, consts.CARD_MIN_DISTANCE)
cv2.drawContours(src, [bounding_box(cnt) for cnt in contours], -1, consts.BOUNDING_BOX_COLOR, 60)
plt.imshow(src)
plt.show()
