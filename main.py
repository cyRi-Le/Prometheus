from game import Game
# en fonction de ton rep
path = "data/train_games/game3"
game = Game(path)
images = game.load_images()
while not game.is_done:
    ROI, dst = game.next_step()
    # ROI contient les cartes des joueurs
    # le model de ML est cense etre appele ici sur ROI
