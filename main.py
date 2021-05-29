import cv2
from game import Game
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

#id = 1  # np.random.randint(1, 5)
#path = f"data/train_games/game{id}"
#res = Path(f"results/game{id}")
#res.mkdir(exist_ok=True)
#game = Game(path)
#images = game.load_images()
#k = 1
for j in range(1, 6):
    path = f"data/train_games/game{j}"
    game = Game(path)
    game.load_images()
    k = 1
    while not game.is_done:
        ROI, dst = game.next_step()
        #path = res / f"{k}.jpeg"

        res = Path(f"ROI/game{j}")
        res.mkdir(exist_ok=True)
        for i, roi in enumerate(ROI):
            #image = roi
            cv2.imwrite(str((res/f"P{i+1} {k}.jpeg").absolute()), roi)
        k += 1
    # break

