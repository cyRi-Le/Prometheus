import cv2
from game import Game
from pathlib import Path
import matplotlib.pyplot as plt

id = 3
path = f"data/train_games/game{id}"
res = Path(f"results/game{id}")
res.mkdir()
game = Game(path)
images = game.load_images()
k = 1
while not game.is_done:
    ROI, dst = game.next_step()
    path = res/f"{k}.jpeg"
    cv2.imwrite(str(path.absolute()), dst)
    k += 1
