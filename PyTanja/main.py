import traceback
import pygame

from pygame import mixer

from game import Game

try:
    pygame.init()

    mixer.init()
    mixer.music.load("in_the_end.mp3")
    mixer.music.play()

    g = Game()
    g.run()

except (Exception,):
    traceback.print_exc()
    input()
finally:
    pygame.quit()
