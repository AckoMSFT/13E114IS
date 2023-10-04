import os
import sys
import pygame
import config
from sprites import Stone, Grass, Dune, Water, Road, Mud, Goal, Trail


class EndGame(Exception):
    pass


class Game:
    def __init__(self):
        self.path_cost = 0
        pygame.display.set_caption('PyTanja')
        values = Game.load_map(sys.argv[1] if len(sys.argv) > 1 else os.path.join(config.MAP_FOLDER, 'map0.txt'))
        self.char_map = values[0]
        self.start = values[1:3]
        self.goal = values[3:]
        # window scaling
        config.TILE_SIZE = min(config.MAX_HEIGHT // len(self.char_map), config.MAX_WIDTH // len(self.char_map[0]))
        config.HEIGHT = config.TILE_SIZE * len(self.char_map)
        config.WIDTH = config.TILE_SIZE * len(self.char_map[0])
        config.GAME_SPEED = int(config.TILE_SIZE * 2)
        pygame.font.init()
        config.GAME_FONT = pygame.font.Font(None, config.TILE_SIZE // 3)
        config.RIBBON_HEIGHT = int(config.GAME_FONT.size('')[1] * 1.5)
        self.screen = pygame.display.set_mode((config.WIDTH, config.HEIGHT + config.RIBBON_HEIGHT))
        self.tiles_sprites = pygame.sprite.Group()
        self.trails_sprites = pygame.sprite.Group()
        self.agents_sprites = pygame.sprite.Group()
        tile_map = []
        for i, row in enumerate(self.char_map):
            map_row = []
            for j, el in enumerate(row):
                if el == 's':
                    t = Stone(i, j)
                elif el == 'w':
                    t = Water(i, j)
                elif el == 'r':
                    t = Road(i, j)
                elif el == 'g':
                    t = Grass(i, j)
                elif el == 'm':
                    t = Mud(i, j)
                elif el == 'd':
                    t = Dune(i, j)
                else:
                    t = Grass(i, j)
                self.tiles_sprites.add(t)
                map_row.append(t)
            tile_map.append(map_row)
        self.tile_map = tile_map
        self.tiles_sprites.add(Goal(self.goal[0], self.goal[1]))
        module = __import__('sprites')
        class_ = getattr(module, sys.argv[2] if len(sys.argv) > 2 else 'ExampleAgent')
        self.agent = class_(self.start[0], self.start[1],
                            f'{sys.argv[2]}.png' if len(sys.argv) > 2 else 'ExampleAgent.png')
        self.agents_sprites.add(self.agent)
        self.clock = pygame.time.Clock()
        self.running = True
        self.playing = False
        self.game_over = False

    @staticmethod
    def load_map(map_name):
        try:
            with open(map_name, 'r') as f:
                ar, ac = [int(val) for val in f.readline().strip().split(',')]
                gr, gc = [int(val) for val in f.readline().strip().split(',')]
                matrix = []
                while True:
                    line = f.readline().strip()
                    if not len(line):
                        break
                    matrix.append([c for c in line])
            return matrix, ar, ac, gr, gc
        except Exception as e:
            raise e

    def check_move(self, old_x, old_y, x, y):
        if abs(old_x - x) + abs(old_y - y) != 1:
            raise Exception(f'ERR: Path nodes {old_x, old_y} and {x, y} are not adjacent!')
        if not (x in range(len(self.tile_map)) and y in range(len(self.tile_map[0]))):
            raise Exception(f'ERR: Agent {x, y} is out of bounds! '
                            f'{len(self.tile_map), len(self.tile_map[0])}')

    def run(self):
        # game loop - set self.playing = False to end the game
        path = self.agent.get_agent_path(self.tile_map, self.goal)
        orig_path = [p for p in path]
        print(f"Path: {', '.join([str(p.position()) for p in path])}")
        print(f'Path length: {len(path)}')
        print(f'Path cost: {sum([t.cost() for t in path])}')
        tile = path.pop(0)
        x, y = tile.position()
        self.path_cost = tile.cost()
        step_count = 1
        game_time = 0
        while self.running:
            try:
                if self.playing:
                    if not game_time:
                        self.agent.place_to(x, y)
                        self.trails_sprites.add(Trail(x, y, step_count))
                        step_count += 1
                        try:
                            tile = path.pop(0)
                        except IndexError:
                            raise EndGame()
                        old_x, old_y = x, y
                        x, y = tile.position()
                        self.check_move(old_x, old_y, x, y)
                        self.path_cost += tile.cost()
                    game_time += 1
                    if game_time == config.TILE_SIZE:
                        game_time = 0
                    self.agent.move_towards(x, y)
                    self.clock.tick(config.GAME_SPEED)
                self.events()
                self.draw()
            except EndGame:
                self.game_over = True
                self.playing = False
                if len(orig_path):
                    self.path_cost = sum([t.cost() for t in orig_path])
                    goal_x, goal_y = orig_path[-1].position()
                    self.trails_sprites = pygame.sprite.Group()
                    for num, tile in enumerate(orig_path):
                        old_x, old_y = x, y
                        x, y = tile.position()
                        if num:
                            self.check_move(old_x, old_y, x, y)
                        self.trails_sprites.add(Trail(x, y, num + 1))
                    self.agent.place_to(goal_x, goal_y)
            except Exception as e:
                self.game_over = True
                raise e

    def quit(self):
        self.running = False

    def draw(self):
        self.screen.fill(config.BLACK, rect=(0, config.HEIGHT, config.WIDTH, config.RIBBON_HEIGHT))
        self.tiles_sprites.draw(self.screen)
        self.trails_sprites.draw(self.screen)
        for t in self.trails_sprites:
            t.draw(self.screen)
        self.agents_sprites.draw(self.screen)
        cost = config.GAME_FONT.render(f'Score: {str(self.path_cost)}', True, config.GREEN)
        self.screen.blit(cost, (10, config.HEIGHT + config.RIBBON_HEIGHT // 5))
        if self.game_over:
            game_over = config.GAME_FONT.render('GAME OVER', True, config.RED)
            text_rect = game_over.get_rect(center=(config.WIDTH // 2, config.HEIGHT // 2))
            self.screen.blit(game_over, text_rect)
        pygame.display.flip()

    def events(self):
        # catch all events here
        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.quit()
            if self.game_over:
                return
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.playing = not self.playing
            elif event.type == pygame.KEYDOWN and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                raise EndGame()
