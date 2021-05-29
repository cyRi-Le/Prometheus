import cv2
import math
import numpy as np
import pandas as pd
from game import Game
from pathlib import Path
from utils import evaluate_game
import matplotlib.pyplot as plt
from models.detector import Detector
from utils import (compute_standard_points, compute_advanced_points,
                   preds_to_ranks, print_results, save_img_files)



idx = np.random.randint(1, 7)
path = "data/train_games/game%d" % idx
game = Game(path)
images = game.load_images()
model = Detector()
pred_dealer = []
pred_rank = []
pred_pts_stand = np.zeros((1, 4), dtype=int)
pred_pts_advan = np.zeros((1, 4), dtype=int)
while not game.is_done:
    ROI, dst = game.next_step(show_step=False, save_fig=True)
    preds = model.prediction_step(ROI)
    dealer_idx = game.dealer_idx
    pred_dealer.append(dealer_idx)
    pred_rank.append(preds.tolist())
    ranks = preds_to_ranks(preds)
    pred_pts_stand += compute_standard_points(preds)
    pred_pts_advan += compute_advanced_points(preds, dealer_idx)

pred_rank = np.asarray(pred_rank)
pred_dealer = np.asarray(pred_dealer)

print_results(
    rank_colour=pred_rank,
    dealer=pred_dealer,
    pts_standard=pred_pts_stand,
    pts_advanced=pred_pts_advan,
)

# Load ground truth from game 1
cgt = pd.read_csv('data/train_games/game%d/game%d.csv' % (idx, idx), index_col=0)
cgt_rank = cgt[['P1', 'P2', 'P3', 'P4']].values
# Compute accuracy of prediction
acc_standard = evaluate_game(pred_rank, cgt_rank, mode_advanced=False)
acc_advanced = evaluate_game(pred_rank, cgt_rank, mode_advanced=True)
print("Your model accuracy is: Standard={:.3f}, Advanced={:.3f}".format(acc_standard, acc_advanced))
res = game.saved_fig
fig = plt.figure(frameon=False)
ax = fig.gca()
ax.axis("off")
ax.imshow(cv2.cvtColor(res[0], cv2.COLOR_BGR2RGB))
plt.show()
save_img_files(res)
