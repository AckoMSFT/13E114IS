import pygame
import os
import config


class BaseSprite(pygame.sprite.Sprite):
    images = dict()

    def __init__(self, row, col, file_name, transparent_color=None):
        pygame.sprite.Sprite.__init__(self)
        if file_name in BaseSprite.images:
            self.image = BaseSprite.images[file_name]
        else:
            self.image = pygame.image.load(os.path.join(config.IMG_FOLDER, file_name)).convert()
            self.image = pygame.transform.scale(self.image, (config.TILE_SIZE, config.TILE_SIZE))
            BaseSprite.images[file_name] = self.image
        # making the image transparent (if needed)
        if transparent_color:
            self.image.set_colorkey(transparent_color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (col * config.TILE_SIZE, row * config.TILE_SIZE)
        self.row = row
        self.col = col


class Agent(BaseSprite):
    def __init__(self, row, col, file_name):
        super(Agent, self).__init__(row, col, file_name, config.DARK_GREEN)

    def move_towards(self, row, col):
        row = row - self.row
        col = col - self.col
        self.rect.x += col
        self.rect.y += row

    def place_to(self, row, col):
        self.row = row
        self.col = col
        self.rect.x = col * config.TILE_SIZE
        self.rect.y = row * config.TILE_SIZE

    # game_map - list of lists of elements of type Tile
    # goal - (row, col)
    # return value - list of elements of type Tile
    def get_agent_path(self, game_map, goal):
        pass


class ExampleAgent(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = [game_map[self.row][self.col]]

        row = self.row
        col = self.col
        while True:
            if row != goal[0]:
                row = row + 1 if row < goal[0] else row - 1
            elif col != goal[1]:
                col = col + 1 if col < goal[1] else col - 1
            else:
                break
            path.append(game_map[row][col])
        return path


dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

from collections import deque as queue
from heapq import heapify, heappush, heappop

def is_out_of_bounds(row, col, height, width):
    if row < 0 or row >= height or col < 0 or col >= width:
        return True
    else:
        return False

"""
Class node which represents a node in the grid
"""
class Node(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col

    def __repr__(self):
        return f'({self.row}, {self.col})'

    def __lt__(self, other):
        return self.row < other.row

    def get_neighbours_average_cost(self, game_map):
        height = len(game_map)
        width = len(game_map[0])
        neighbours_cost = []
        for (idx, idy) in zip(dx, dy):
            row = self.row + idx
            col = self.col + idy
            if is_out_of_bounds(row, col, height, width):
                continue
            neighbours_cost.append(game_map[row][col].cost())

        return sum(neighbours_cost) / len(neighbours_cost)

"""
Class Path defines one path from the root node
"""
class Path(object):
    def __init__(self, game_map, nodes):
        self.game_map = game_map
        self.nodes = queue(nodes)
        self.node_count = len(nodes)
        self.path_cost = sum(self.game_map[node.row][node.col].cost() for node in self.nodes)

    def __repr__(self):
        return " -> ".join(str(node) for node in self.nodes) + " PathCost = " + str(
            self.path_cost) + " NodeCount = " + str(self.node_count)

    def __lt__(self, other):
        if self.path_cost != other.path_cost:
            return self.path_cost < other.path_cost

        return self.node_count < other.node_count

    def add_node(self, node):
        self.nodes.append(node)
        self.node_count += 1
        self.path_cost += self.game_map[node.row][node.col].cost()

    def pop_node(self):
        node = self.nodes[-1]
        self.nodes.pop()
        self.node_count -= 1
        self.path_cost -= self.game_map[node.row][node.col].cost()

    def node_count(self):
        return len(self.nodes)

    def get_path(self):
        return list(map(lambda node: self.game_map[node.row][node.col], self.nodes))

    def get_node_count(self):
        return self.node_count

    def get_path_cost(self):
        return self.path_cost

    def get_last_node(self):
        return self.nodes[-1]

class Neighbour(object):
    def __init__(self, game_map, row, col, priority, agent_row, agent_col):
        self.game_map = game_map
        self.height = len(game_map)
        self.width = len(game_map[0])
        self.row = row
        self.col = col
        self.priority = priority
        self.agent_row = agent_row
        self.agent_col = agent_col

    def __repr__(self):
        return f"({self.row}, {self.col})"

    def __lt__(self, other):
        cost = self.get_average_cost()
        other_cost = other.get_average_cost()

        if cost != other_cost:
            return cost < other_cost

        return self.priority < other.priority

    def get_average_cost(self):
        neighbour_count = 0
        neighbour_cost = 0

        for (idx, idy) in zip(dx, dy):
            neighbour_row = self.row + idx
            neighbour_col = self.col + idy

            if is_out_of_bounds(neighbour_row, neighbour_col, self.height, self.width):
                continue

            if neighbour_row == self.agent_row and neighbour_col == self.agent_col:
                continue

            neighbour_count += 1
            neighbour_cost += self.game_map[neighbour_row][neighbour_col].cost()

        return neighbour_cost / neighbour_count if neighbour_count > 0 else 0


def depth_breath_first_search(game_map, start_row, start_col, end_row, end_col, is_depth_first_search):
    # Calculate width and height of the map
    height = len(game_map)
    width = len(game_map[0])

    # Initialize the visited matrix which tells us if a certain node has been visited or not
    visited = [[False for j in range(width)] for i in range(height)]
    visited[start_row][start_col] = True

    # Initialize the previous matrix which tells us from which node we reached a given node
    previous = [[(-1, -1) for j in range(width)] for i in range(height)]

    # Initialize list of nodes for search and add the starting node
    nodes = queue()
    nodes.append((start_row, start_col))
    visited[start_row][start_col] = True

    # While there are nodes to search from
    while (len(nodes) > 0):
        # Get the current node
        (current_row, current_col) = nodes.popleft()

        # Find all of the neighbours of the current node
        neighbours = []
        priority = 0
        for (idx, idy) in zip(dx, dy):
            row = current_row + idx
            col = current_col + idy

            # If we are out of bounds of the grid we continue
            if is_out_of_bounds(row, col, height, width):
                continue

            # If we have seen this node we continue
            if visited[row][col] == True:
                continue

            # Priority tells us the priority for the agents in case of a tie - north / east / south / west
            previous[row][col] = (current_row, current_col)
            neighour = Neighbour(game_map, row, col, priority, current_row, current_col)

            neighbours.append(neighour)
            priority = priority + 1

        # Sort the neighbours and reverse the list if we are performing DFS
        neighbours.sort()
        if is_depth_first_search:
            neighbours.reverse()

        # Iterate through the neighbours
        for neighbour in neighbours:
            row = neighbour.row
            col = neighbour.col

            # Mark the current node as visited and add to the front in case of DFS or to the back in case of BFS
            visited[row][col] = True
            if is_depth_first_search:
                nodes.appendleft((row, col))
            else:
                nodes.append((row, col))

    # Reconstruct the path starting from the end node
    row = end_row
    col = end_col
    path = queue()
    while True:
        # No need to go further if we have reached the start node
        if row == start_row and col == start_col:
            break;

        path.appendleft(Node(row, col))
        (row, col) = previous[row][col]

    path.appendleft(Node(row, col))

    return Path(game_map, path)


def branch_and_bound_search(game_map, start_row, start_col, end_row, end_col):
    # Initialize the min heap with starting path containing only the starting node
    branch_and_bound_min_heap = [Path(game_map, [Node(start_row, start_col)])]
    heapify(branch_and_bound_min_heap)

    # Calculate width and height of the map
    height = len(game_map)
    width = len(game_map[0])

    # Initialize the visited matrix which tells us if a certain node has been visited or not
    visited = [[False for j in range(width)] for i in range(height)]
    visited[start_row][start_col] = True

    # Loop until we find the path to the goal
    while True:
        # Get the best partial path we have until now
        current_path = heappop(branch_and_bound_min_heap)
        last_node = current_path.get_last_node()

        # print("CurrentPath: " + str(current_path))

        # Iterate through all 4 neighbours of the last node on the path
        for (idx, idy) in zip(dx, dy):
            # Calculate the neighbour, make sure we are not out of bounds and that we haven't seen this node yet
            (row, col) = (last_node.row + idx, last_node.col + idy)
            if is_out_of_bounds(row, col, height, width):
                continue

            if visited[row][col]:
                continue
            visited[row][col] = True

            # Add the neighbour to the path
            next_path = Path(game_map, current_path.nodes)
            next_path.add_node(Node(row, col))

            # Woo-hoo! We have reached our goal! :)
            if row == end_row and col == end_col:
                return next_path

            # print("NextPath: " + str(next_path))

            # Add the path to the heap maintaining our best partial paths
            heappush(branch_and_bound_min_heap, next_path)

def a_star(game_map, start_row, start_col, end_row, end_col):
    # Initialize the min heap with starting path containing only the starting node
    branch_and_bound_min_heap = [Path(game_map, [Node(start_row, start_col)])]
    heapify(branch_and_bound_min_heap)

    # Calculate width and height of the map
    height = len(game_map)
    width = len(game_map[0])

    # Initialize the visited matrix which tells us if a certain node has been visited or not
    visited = [[False for j in range(width)] for i in range(height)]
    visited[start_row][start_col] = True

    # Loop until we find the path to the goal
    while True:
        # Get the best partial path we have until now
        current_path = heappop(branch_and_bound_min_heap)
        last_node = current_path.get_last_node()

        # print("CurrentPath: " + str(current_path))

        # Iterate through all 4 neighbours of the last node on the path
        for (idx, idy) in zip(dx, dy):
            # Calculate the neighbour, make sure we are not out of bounds and that we haven't seen this node yet
            (row, col) = (last_node.row + idx, last_node.col + idy)
            if is_out_of_bounds(row, col, height, width):
                continue

            if visited[row][col]:
                continue
            visited[row][col] = True

            # Add the neighbour to the path
            next_path = Path(game_map, current_path.nodes)
            next_path.add_node(Node(row, col))

            # Woo-hoo! We have reached our goal! :)
            if row == end_row and col == end_col:
                return next_path

            # print("NextPath: " + str(next_path))

            # Heuristic will be a cost of a path from the end node to the current row using branch and bound search
            heuristic = branch_and_bound_search(game_map, end_row, end_col, row, col)
            cost = heuristic.get_path_cost()

            next_path.path_cost += cost

            # Add the path to the heap maintaining our best partial paths
            heappush(branch_and_bound_min_heap, next_path)

class Aki(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = depth_breath_first_search(game_map, self.row, self.col, goal[0], goal[1], True)
        return path.get_path()


class Jocke(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = depth_breath_first_search(game_map, self.row, self.col, goal[0], goal[1], False)
        return path.get_path()


class Draza(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = branch_and_bound_search(game_map, self.row, self.col, goal[0], goal[1])
        return path.get_path()

class Bole(Agent):
    def __init__(self, row, col, file_name):
        super().__init__(row, col, file_name)

    def get_agent_path(self, game_map, goal):
        path = a_star(game_map, self.row, self.col, goal[0], goal[1])
        return path.get_path()

class Tile(BaseSprite):
    def __init__(self, row, col, file_name):
        super(Tile, self).__init__(row, col, file_name)

    def position(self):
        return self.row, self.col

    def cost(self):
        pass

    def kind(self):
        pass


class Stone(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'stone.png')

    def cost(self):
        return 1000

    def kind(self):
        return 's'


class Water(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'water.png')

    def cost(self):
        return 500

    def kind(self):
        return 'w'


class Road(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'road.png')

    def cost(self):
        return 2

    def kind(self):
        return 'r'


class Grass(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'grass.png')

    def cost(self):
        return 3

    def kind(self):
        return 'g'


class Mud(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'mud.png')

    def cost(self):
        return 5

    def kind(self):
        return 'm'


class Dune(Tile):
    def __init__(self, row, col):
        super().__init__(row, col, 'dune.png')

    def cost(self):
        return 7

    def kind(self):
        return 's'


class Goal(BaseSprite):
    def __init__(self, row, col):
        super().__init__(row, col, 'x.png', config.DARK_GREEN)


class Trail(BaseSprite):
    def __init__(self, row, col, num):
        super().__init__(row, col, 'trail.png', config.DARK_GREEN)
        self.num = num

    def draw(self, screen):
        text = config.GAME_FONT.render(f'{self.num}', True, config.WHITE)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)
