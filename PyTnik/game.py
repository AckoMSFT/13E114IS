import logging
import os
import sys
import threading
import time
from queue import Queue

import pygame

import config
from sprites import Coin, CollectedCoin, Surface
from util import TimedFunction, Timeout


class EndGame(Exception):
    pass


class Game:
    @staticmethod
    def load_map(map_name):
        try:
            with open(map_name, 'r') as f:
                ax, ay = [int(val) for val in f.readline().strip().split(',')[:2]]
                coin_distance = [[0]]
                coins_sprites = pygame.sprite.Group()
                coins = []
                coin = Coin(ax, ay, 0)
                coins_sprites.add(coin)
                coins.append(coin)
                ident = 1
                while True:
                    line = f.readline().strip()
                    if not len(line):
                        break
                    values = [int(val) for val in line.split(',')]
                    cx, cy = values[:2]
                    coin = Coin(cx, cy, ident)
                    coins_sprites.add(coin)
                    coins.append(coin)
                    ident += 1
                    for iteration, coin_sublist in enumerate(coin_distance):
                        coin_sublist.append(values[2 + iteration])
                    coin_distance.append(values[2:2 + len(coin_distance)] + [0])
                return (ax, ay), coin_distance, coins, coins_sprites
        except Exception as e:
            raise e

    def __init__(self):
        pygame.display.set_caption('Pytnik')
        pygame.font.init()
        config.GAME_FONT = pygame.font.Font(os.path.join(config.FONT_FOLDER, 'game_font.ttf'), 40)
        config.INFO_FONT = pygame.font.Font(os.path.join(config.FONT_FOLDER, 'info_font.ttf'), 16)
        config.COIN_FONT = pygame.font.Font(os.path.join(config.FONT_FOLDER, 'coin_font.ttf'), config.SPRITE_SIZE // 3)
        self.screen = pygame.display.set_mode((config.WIDTH + config.SIDE_WIDTH, config.HEIGHT))
        self.surface_sprite = pygame.sprite.Group()
        self.surface_sprite.add(Surface())
        agent_pos, self.coin_distance, self.coins, self.coins_sprites = Game.load_map(
            sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.MAP_FOLDER, 'map0.txt'))
        self.collected_coins = [CollectedCoin(coin) for coin in self.coins]
        self.collected_coins_sprites = pygame.sprite.Group()
        module = __import__('sprites')
        class_ = getattr(module, sys.argv[2] if len(sys.argv) > 2 else 'ExampleAgent')
        self.agent = class_(agent_pos[0], agent_pos[1],
                            f'{sys.argv[2]}.png' if len(sys.argv) > 2 else 'ExampleAgent.png',
                            float(sys.argv[3]) if len(sys.argv) > 3 else 5.)
        self.max_elapsed_time = float(sys.argv[3]) if len(sys.argv) > 3 else 5.
        self.elapsed_time = 0.
        self.agent_sprites = pygame.sprite.Group()
        self.agent_sprites.add(self.agent)
        self.clock = pygame.time.Clock()
        self.nodes = None
        self.current_path = None
        self.current_path_cost = None
        self.running = True
        self.playing = False
        self.game_over = False
        self.stepping = False
        self.moving = False
        self.direction = 1
        self.time_out = False
        self.proper_path = True

    def run(self):
        self.nodes = None
        self.current_path = []
        self.current_path_cost = 0
        from_id, to_id = None, None
        journey_start = True
        while self.running:
            try:
                try:
                    if self.nodes is None and not self.time_out:
                        self.draw()
                        tf_queue = Queue(1)
                        tf = TimedFunction(threading.current_thread().ident, tf_queue,
                                           self.max_elapsed_time, self.agent.get_agent_path, self.coin_distance)
                        tf.daemon = True
                        tf.start()
                        start_time = time.time()
                        sleep_time = 0.001
                        while tf_queue.empty():
                            time.sleep(sleep_time)
                            self.elapsed_time = time.time() - start_time
                            self.draw_time_meter()
                            self.draw_calculating_text()
                            self.events()
                        self.nodes, elapsed = tf_queue.get(block=False)
                        if not (min(self.nodes) == 0
                                and max(self.nodes) == len(self.nodes) - 2 == len(set(self.nodes)) - 1
                                and self.nodes[0] == self.nodes[-1] == 0):
                            print(f'ERR: Path {self.nodes} is not a permutation 0-N or does not start or end with 0!')
                            self.proper_path = False
                            raise EndGame()
                        print(f'Algorithm time elapsed: {elapsed:.3f} seconds.')
                except Timeout:
                    print(f'ERR: Algorithm took more than {self.max_elapsed_time} seconds!')
                    self.time_out = True
                    raise EndGame

                if self.playing:
                    if not self.agent.is_travelling() and journey_start and (not self.stepping or self.moving):
                        try:
                            from_id = self.nodes[len(self.current_path)]
                            to_id = self.nodes[len(self.current_path) + self.direction]
                        except IndexError:
                            raise EndGame()
                        x, y = self.coins[to_id].position()
                        self.agent.set_destination(x, y)
                        if self.moving:
                            self.moving = False
                        journey_start = False
                    self.agent.move_one_step()
                    if not self.agent.is_travelling() and not journey_start:
                        cost = self.coin_distance[from_id][to_id]
                        self.current_path_cost += cost if self.direction == 1 else -cost
                        if self.direction == 1:
                            self.current_path.append((from_id, to_id, self.coin_distance[from_id][to_id]))
                            coin = self.coins[to_id]
                            self.coins_sprites.remove(coin)
                            self.collected_coins_sprites.add(self.collected_coins[to_id])
                        else:
                            del self.current_path[-1]
                            coin = self.coins[from_id]
                            self.coins_sprites.add(coin)
                            self.collected_coins_sprites.remove(self.collected_coins[from_id])
                        self.direction = 1
                        journey_start = True
                    self.clock.tick(config.FRAME_RATE)
                self.events()
                self.draw()
            except EndGame:
                self.game_over = True
                self.playing = False
                if self.nodes is not None and self.proper_path:
                    remove_ids = [coin.get_ident() if hasattr(coin, 'get_ident') else -1 for coin in self.coins_sprites]
                    for ident in remove_ids:
                        self.coins_sprites.remove(self.coins[ident])
                        self.collected_coins_sprites.add(self.collected_coins[ident])
                    self.agent.place_to(self.coins[0].position())
                    self.current_path.clear()
                    self.current_path_cost = 0
                    for i in range(len(self.nodes) - 1):
                        from_id, to_id = self.nodes[i], self.nodes[i + 1]
                        self.current_path.append((from_id, to_id, self.coin_distance[from_id][to_id]))
                        self.current_path_cost += self.coin_distance[from_id][to_id]
            except Exception as e:
                raise e

    def draw_time_meter(self):
        x, y, w, h, m = 20, 20, 120, 30, 3
        color = config.BLACK
        self.screen.fill(color, rect=(x, y, w, h))
        perc_left = max(int((1 - self.elapsed_time / self.max_elapsed_time) * 100), 0)
        self.screen.fill(config.R_to_G[perc_left], rect=(x + m, y + m, w * perc_left * 0.01 - 2 * m, h - 2 * m))
        time_left = max(self.max_elapsed_time - self.elapsed_time, 0)
        text = f'{time_left:.3f}s'
        text_width, text_height = config.INFO_FONT.size(text)
        text = config.INFO_FONT.render(text, True, config.WHITE)
        self.screen.blit(text, (x + w // 2 + m // 2 - text_width // 2, y + h // 2 - m // 2 - text_height // 2))
        pygame.display.flip()

    def draw_path(self):
        self.screen.fill(config.BLACK, rect=(config.WIDTH, 0, config.SIDE_WIDTH, config.HEIGHT))
        text = f'======= Steps ======='
        _, text_height = config.INFO_FONT.size(text)
        left = config.WIDTH + 10
        text = config.INFO_FONT.render(text, True, config.GREEN)
        self.screen.blit(text, (left + 5, 10))
        s_ind = 0 if len(self.current_path) < 20 else len(self.current_path) - 20
        for i, part in enumerate(self.current_path[s_ind:]):
            text = config.INFO_FONT.render(f'{(i + s_ind + 1):2} | '
                                           f'{part[0]:3} - {part[1]:3} : ', True, config.GREEN)
            top = int(5 + text_height * 1.5 * (i + 1))
            self.screen.blit(text, (left, top))
            text = config.INFO_FONT.render(f'{part[2]:3}', True, config.WHITE)
            self.screen.blit(text, (left + 180, top))
        text = config.INFO_FONT.render('=' * 22, True, config.GREEN)
        self.screen.blit(text, (left, config.HEIGHT - 50))
        text = config.INFO_FONT.render(f'Cost: {self.current_path_cost}', True, config.GREEN)
        self.screen.blit(text, (left, config.HEIGHT - 30))

    def draw_calculating_text(self):
        if 'subsurface' not in Game.draw_calculating_text.__dict__:
            text_width, text_height = config.GAME_FONT.size('CALCULATING ...')
            Game.draw_calculating_text.rect = (config.WIDTH // 2 - text_width // 2,
                                               config.HEIGHT // 2 - text_height // 2, text_width, text_height)
            Game.draw_calculating_text.subsurface = self.screen.subsurface(Game.draw_calculating_text.rect).copy()

        dots = (int(self.elapsed_time * 4) % 4)
        text = config.GAME_FONT.render(f'CALCULATING {"." * dots}', True, config.YELLOW)
        text.set_alpha([159, 191, 223, 255][dots])
        self.screen.blit(Game.draw_calculating_text.subsurface, Game.draw_calculating_text.rect)
        self.screen.blit(text, Game.draw_calculating_text.rect[:2])

    def draw_info_text(self):
        text = 'TIMED OUT' if self.time_out \
            else '' if self.nodes is None \
            else 'GAME OVER' if self.game_over \
            else 'PAUSED'
        if len(text):
            text_width, text_height = config.GAME_FONT.size(text)
            text = config.GAME_FONT.render(text, True, config.RED)
            self.screen.blit(text, (config.WIDTH // 2 - text_width // 2, config.HEIGHT // 2 - text_height // 2))

    def draw_step_text(self):
        text = f'STEP {len(self.current_path)}/{len(self.nodes) - 1}'
        text_width, _ = config.GAME_FONT.size(text)
        text = config.GAME_FONT.render(text, True, config.WHITE)
        self.screen.blit(text, (config.WIDTH // 2 - text_width // 2, 10))

    def draw(self):
        self.surface_sprite.draw(self.screen)
        self.coins_sprites.draw(self.screen)
        for coin in self.coins_sprites:
            if hasattr(coin, 'draw'):
                coin.draw(self.screen)
        self.collected_coins_sprites.draw(self.screen)
        for coin in self.collected_coins_sprites:
            if hasattr(coin, 'draw'):
                coin.draw(self.screen)
        self.agent_sprites.draw(self.screen)
        self.draw_path()
        if not self.playing:
            self.draw_info_text()
        if self.stepping:
            self.draw_step_text()
        pygame.display.flip()

    def events(self):
        # catch all events here
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.WINDOWCLOSE or \
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.running = False
                raise EndGame
            if self.game_over or self.nodes is None:
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.playing = not self.playing
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                self.stepping = not self.stepping
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RIGHT and \
                    self.stepping and self.playing and not self.agent.is_travelling() \
                    and len(self.current_path) < len(self.nodes) - 1:
                self.moving = True
                self.direction = 1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_LEFT and \
                    self.stepping and self.playing and not self.agent.is_travelling() and len(self.current_path) > 0:
                self.moving = True
                self.direction = -1
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                raise EndGame()
