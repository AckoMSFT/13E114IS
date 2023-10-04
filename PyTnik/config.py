import os

# parameters
HEIGHT = 680
WIDTH = 1080
SIDE_WIDTH = 240
SPRITE_SIZE = 64
FRAME_RATE = 60
TRAVEL_SPEED = 10
GAME_FONT = None
INFO_FONT = None
COIN_FONT = None
RIBBON_HEIGHT = None

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (192, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 128, 0)
YELLOW = (255, 255, 0)

GR_LEN = 101
R_to_G = [((255 * (100 - i)) / 100, (255 * i) / 100, 0) for i in range(GR_LEN)]

GAME_FOLDER = os.path.dirname(__file__)
IMG_FOLDER = os.path.join(GAME_FOLDER, 'img')
MAP_FOLDER = os.path.join(GAME_FOLDER, 'maps')
FONT_FOLDER = os.path.join(GAME_FOLDER, 'fonts')