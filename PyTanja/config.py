import os

# parameters
MAX_HEIGHT = 600
MAX_WIDTH = 1300
WIDTH = None
HEIGHT = None
TILE_SIZE = None
GAME_SPEED = None
GAME_FONT = None
RIBBON_HEIGHT = None

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 128, 0)

GAME_FOLDER = os.path.dirname(__file__)
IMG_FOLDER = os.path.join(GAME_FOLDER, 'img')
MAP_FOLDER = os.path.join(GAME_FOLDER, 'maps')
