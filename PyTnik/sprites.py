from collections import deque as queue
from heapq import heapify, heappush, heappop
from itertools import permutations

import math
import random
import sys
import time

import pygame
import os
import config


class BaseSprite(pygame.sprite.Sprite):
    images = dict()

    def __init__(self, x, y, file_name, transparent_color=None, wid=config.SPRITE_SIZE, hei=config.SPRITE_SIZE):
        pygame.sprite.Sprite.__init__(self)
        if file_name in BaseSprite.images:
            self.image = BaseSprite.images[file_name]
        else:
            self.image = pygame.image.load(os.path.join(config.IMG_FOLDER, file_name)).convert()
            self.image = pygame.transform.scale(self.image, (wid, hei))
            BaseSprite.images[file_name] = self.image
        # making the image transparent (if needed)
        if transparent_color:
            self.image.set_colorkey(transparent_color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)


class Surface(BaseSprite):
    def __init__(self):
        super(Surface, self).__init__(0, 0, 'terrain.png', None, config.WIDTH, config.HEIGHT)


class Coin(BaseSprite):
    def __init__(self, x, y, ident):
        self.ident = ident
        super(Coin, self).__init__(x, y, 'coin.png', config.DARK_GREEN)

    def get_ident(self):
        return self.ident

    def position(self):
        return self.rect.x, self.rect.y

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.BLACK)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class CollectedCoin(BaseSprite):
    def __init__(self, coin):
        self.ident = coin.ident
        super(CollectedCoin, self).__init__(coin.rect.x, coin.rect.y, 'collected_coin.png', config.DARK_GREEN)

    def draw(self, screen):
        text = config.COIN_FONT.render(f'{self.ident}', True, config.RED)
        text_rect = text.get_rect(center=self.rect.center)
        screen.blit(text, text_rect)


class Agent(BaseSprite):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super(Agent, self).__init__(x, y, file_name, config.DARK_GREEN)
        self.x = self.rect.x
        self.y = self.rect.y
        self.step = None
        self.travelling = False
        self.destinationX = 0
        self.destinationY = 0
        self.max_elapsed_time = max_elapsed_time

    def set_destination(self, x, y):
        self.destinationX = x
        self.destinationY = y
        self.step = [self.destinationX - self.x, self.destinationY - self.y]
        magnitude = math.sqrt(self.step[0] ** 2 + self.step[1] ** 2)
        self.step[0] /= magnitude
        self.step[1] /= magnitude
        self.step[0] *= config.TRAVEL_SPEED
        self.step[1] *= config.TRAVEL_SPEED
        self.travelling = True

    def move_one_step(self):
        if not self.travelling:
            return
        self.x += self.step[0]
        self.y += self.step[1]
        self.rect.x = self.x
        self.rect.y = self.y
        if abs(self.x - self.destinationX) < abs(self.step[0]) and abs(self.y - self.destinationY) < abs(self.step[1]):
            self.rect.x = self.destinationX
            self.rect.y = self.destinationY
            self.x = self.destinationX
            self.y = self.destinationY
            self.travelling = False

    def is_travelling(self):
        return self.travelling

    def place_to(self, position):
        self.x = self.destinationX = self.rect.x = position[0]
        self.y = self.destinationX = self.rect.y = position[1]

    # coin_distance - cost matrix
    # return value - list of coin identifiers (containing 0 as first and last element, as well)
    def get_agent_path(self, coin_distance):
        pass


class ExampleAgent(Agent):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super().__init__(x, y, file_name, max_elapsed_time)

    def get_agent_path(self, coin_distance):
        path = [i for i in range(1, len(coin_distance))]
        random.shuffle(path)
        return [0] + path + [0]


'''
    Aki uses greedy DFS search by always collecting the coin which has the cheapest cost of getting it.
    If there are tie-breakers Aki gets the coin with the smallest ID.
'''


class Aki(Agent):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super().__init__(x, y, file_name, max_elapsed_time)

    def get_agent_path(self, coin_distance):
        num_coins = len(coin_distance)

        # visited[i] represents if we collected that coin or not, we mark that coin 0 is collected since that is our starting point
        visited = [False] * num_coins
        visited[0] = True

        # path will represent the final path for the agent
        path = [0]
        # we start form coin 0
        current_coin = 0
        while True:
            idx_min = -1
            min_cost = sys.maxsize

            # from all of the uncollected coins we find the one with the least cost of collecting it
            for next_coin in range(num_coins):
                # we are only interested in uncollected coins
                if visited[next_coin]:
                    continue

                if coin_distance[current_coin][next_coin] < min_cost:
                    min_cost = coin_distance[current_coin][next_coin]
                    idx_min = next_coin

            # if we have found the coin to collect we collect it
            if idx_min != -1:
                current_coin = idx_min
                visited[current_coin] = True
                path = path + [current_coin]
            else:
                break

        # we always end back where we started - coin 0
        path = path + [0]
        return path


'''
    Helper function which calculates the cost of a partial path given the partial path and start and end coins
'''


def calculate_cost(path, cost_distance, start=0, end=0):
    current = start
    cost = 0
    # iterate over the path and calculate the cost
    for node in path:
        cost += cost_distance[current][node]
        current = node
    # add the cost of jumping from the end of the partial path to the end coin
    cost += cost_distance[current][end]
    return cost


'''
    Jocke uses brute force by generating all possible paths and chooses the one with the smallest cost.
'''


class Jocke(Agent):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super().__init__(x, y, file_name, max_elapsed_time)

    def get_agent_path(self, coin_distance):
        num_coins = len(coin_distance)

        # we generate all possible paths - which means all permutations of set {1 ... N} assuming we have N + 1 coins (since we always start and end with 0)
        paths = list(permutations(range(1, num_coins)))
        best_path = []
        best_path_cost = sys.maxsize

        # iterate over all permutations
        for path in paths:
            # calculate the cost of the current permutation
            cost = calculate_cost(path, coin_distance)
            # if the cost is smaller than the current minimal cost update it
            if cost < best_path_cost:
                best_path_cost = cost
                best_path = path

        # we start and end with 0
        path = [0] + list(best_path) + [0]
        return path


"""
    Class Path defines one partial path from the root node where cost is the actual cost of the path
"""


class Path(object):
    def __init__(self, coin_distance, nodes):
        self.coin_distance = coin_distance
        self.nodes = queue(nodes)
        self.node_count = len(nodes)
        self.path_cost = calculate_cost(nodes, coin_distance, 0, self.nodes[-1])

    def __repr__(self):
        return " -> ".join(str(node) for node in self.nodes) + " PathCost = " + str(
            self.path_cost) + " NodeCount = " + str(self.node_count)

    def __lt__(self, other):
        if self.path_cost != other.path_cost:
            return self.path_cost < other.path_cost

        if self.node_count != other.node_count:
            return self.node_count > other.node_count

        return self.get_last_node() < other.get_last_node()

    def get_path(self):
        return self.nodes

    def get_node_count(self):
        return self.node_count

    def get_path_cost(self):
        return self.path_cost

    def get_last_node(self):
        return self.nodes[-1]


'''
    Uki uses branch and bound search. If there are multiple partial paths with the same cost Uki chooses the one with the most collected coins.
    In case of a tie breaker Uki chooses one that leads to the coin with the smallest ID.
'''


class Uki(Agent):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super().__init__(x, y, file_name, max_elapsed_time)

    def get_agent_path(self, coin_distance):
        num_coins = len(coin_distance)

        # Initialize the min heap with starting path containing only the starting node
        branch_and_bound_min_heap = [Path(coin_distance, [0])]
        heapify(branch_and_bound_min_heap)

        best_path = [0]
        # Loop until we find the path to the goal
        while len(branch_and_bound_min_heap) != 0:
            # Get the best partial path we have until now
            current = heappop(branch_and_bound_min_heap)

            if (current.get_node_count() == num_coins + 1):
                best_path = current.get_path()
                break

            path = current.get_path()
            if (current.get_node_count() == num_coins):
                next_node = 0
                next_path = list(path) + [next_node]
                # Add the path to the heap maintaining our best partial paths
                heappush(branch_and_bound_min_heap, Path(coin_distance, next_path))
                continue

            for next_node in range(1, num_coins):
                if next_node in path:
                    continue

                next_path = list(path) + [next_node]
                # Add the path to the heap maintaining our best partial paths
                heappush(branch_and_bound_min_heap, Path(coin_distance, next_path))

        path = list(best_path)
        return path


'''
    Class that represents the partial graph of the partial path where cost is calculated using MST as an A* heuristic
'''


class Graph(object):
    def __init__(self, nodes, coin_distance):
        self.nodes = nodes
        self.node_count = len(coin_distance)
        self.coin_distance = coin_distance
        self.parent = [None] * self.node_count
        self.mst_cost = self.get_mst_cost()

    def __lt__(self, other):
        mst_cost = self.mst_cost + calculate_cost(self.nodes, self.coin_distance, 0, self.nodes[-1])
        other_mst_cost = other.mst_cost + calculate_cost(other.nodes, other.coin_distance, 0, other.nodes[-1])

        node_count = self.get_node_count()
        other_node_count = other.get_node_count()
        last_node = self.get_last_node()
        other_last_node = other.get_last_node()

        if mst_cost != other_mst_cost:
            return mst_cost < other_mst_cost

        if node_count != other_mst_cost:
            return node_count > other_node_count

        return last_node < other_last_node

    def get_node_count(self):
        return len(self.nodes)

    def get_last_node(self):
        return self.nodes[-1]

    def get_nodes(self):
        return self.nodes

    def get_mst_cost(self):
        num_coins = len(self.coin_distance)

        self.parent = [None] * num_coins
        self.children = [None] * num_coins

        min_edges = [sys.maxsize] * num_coins
        min_edges[0] = 0
        in_set = [False] * num_coins
        is_valid = [False] * num_coins

        invalid_nodes = self.nodes[1:-1] if self.node_count > 1 else []

        count_valid = 1
        for node in range(0, num_coins):
            is_valid[node] = not (node in invalid_nodes)
            if is_valid[node]:
                count_valid += 1

        # 0 must always be in partial graph
        self.parent[0] = -1
        for _ in range(count_valid - 1):
            min_edge = sys.maxsize
            idx_min_edge = -1

            for node in range(num_coins):
                if not is_valid[node]:
                    continue

                if in_set[node]:
                    continue

                if min_edges[node] < min_edge:
                    min_edge = min_edges[node]
                    idx_min_edge = node

            in_set[idx_min_edge] = True
            for next_node in range(num_coins):
                if not is_valid[next_node]:
                    continue

                if next_node == idx_min_edge:
                    continue

                if in_set[next_node]:
                    continue

                if min_edges[next_node] > self.coin_distance[idx_min_edge][next_node]:
                    min_edges[next_node] = self.coin_distance[idx_min_edge][next_node]
                    self.parent[next_node] = idx_min_edge

        total_cost = 0
        for node in range(num_coins):
            if is_valid[node]:
                if node != 0:
                    total_cost += self.coin_distance[self.parent[node]][node]
        return total_cost


'''
    Micko uses A* algorithm with minimum spanning tree as the heuristics. If there are multiple partial paths with the same value Micko chooses the one
    with the most collected coins. In case of a tie breaker Micko chooses the one which leads to the coin with the smallest ID.
'''


class Micko(Agent):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super().__init__(x, y, file_name, max_elapsed_time)

    def get_agent_path(self, coin_distance):
        num_coins = len(coin_distance)

        # Initialize the min heap with starting graph containing only the starting node
        branch_and_bound_min_heap = [Graph([0], coin_distance)]
        heapify(branch_and_bound_min_heap)

        best_graph = None

        # Loop until we find the path to the goal
        while True:
            # Get the best partial graph we have until now
            current = heappop(branch_and_bound_min_heap)

            if (current.get_node_count() == num_coins):
                best_graph = current
                break

            for next_node in range(1, num_coins):
                if next_node in current.nodes:
                    continue

                current_nodes = list(current.nodes)
                current_nodes.append(next_node)

                next_graph = Graph(current_nodes, coin_distance)
                heappush(branch_and_bound_min_heap, next_graph)

        best_path = best_graph.nodes
        return best_path + [0]


def two_opt(path, idx, idy):
    # split the path at positions idx and idy and reverse the middle and insert it back
    new_path = list(path[0:idx])
    middle = path[idx : idy + 1]
    middle.reverse()
    new_path.extend(middle)
    new_path.extend(path[idy + 1: ])
    return new_path

'''
    Yasuo uses 2-OPT local search algorithm
'''
class Yasuo(Agent):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super().__init__(x, y, file_name, max_elapsed_time)

    def get_agent_path(self, coin_distance):
        num_coins = len(coin_distance)
        end_time = time.time() + int(self.max_elapsed_time - 1)

        path = [i for i in range(0, num_coins)]
        cost = calculate_cost(path, coin_distance)

        while time.time() < end_time:
            found_improvement = False

            for idx in range(1, num_coins - 1):
                if found_improvement:
                    break

                for idy in range(idx + 1, num_coins - 1):
                    new_path = two_opt(path, idx, idy)
                    new_cost = calculate_cost(new_path, coin_distance)

                    if new_cost < cost:
                        found_improvement = True
                        cost = new_cost
                        path = list(new_path)
                        break

            if found_improvement == False:
                break

        path = path + [0]
        return path


"""
    Break the path at three different edges (a -> b, c -> d, e -> f) and reconstruct them to maintain a complete tour
    There are 8 ways to do so:
        - 1 is identity (we have the same path as before)
        - 3 are 2-opt because we reconstruct one of the original edges
        - 4 are real 3-opt moves
"""
def three_opt(path, allow_two_opt=False):
    num_coins = len(path)

    a, c, e = random.sample(range(num_coins + 1), 3)

    a, c, e = sorted([a, c, e])
    b, d, f = a + 1, c + 1, e + 1

    if allow_two_opt:
        choice = random.randint(0, 7)
    else:
        choice = random.choice([3, 4, 5, 6])

    if choice == 0:
        new_path = path[: a + 1] + path[b:c + 1] + path[d:e + 1] + path[f:]  # identity
    elif choice == 1:
        new_path = path[:a + 1] + path[b:c + 1] + path[e:d - 1:-1] + path[f:]  # 2-opt
    elif choice == 2:
        new_path = path[:a + 1] + path[c:b - 1:-1] + path[d:e + 1] + path[f:]  # 2-opt
    elif choice == 3:
        new_path = path[:a + 1] + path[c:b - 1:-1] + path[e:d - 1:-1] + path[f:]  # 3-opt
    elif choice == 4:
        new_path = path[:a + 1] + path[d:e + 1] + path[b:c + 1] + path[f:]  # 3-opt
    elif choice == 5:
        new_path = path[:a + 1] + path[d:e + 1] + path[c:b - 1:-1] + path[f:]  # 3-opt
    elif choice == 6:
        new_path = path[:a + 1] + path[e:d - 1:-1] + path[b:c + 1] + path[f:]  # 3-opt
    elif choice == 7:
        new_path = path[:a + 1] + path[e:d - 1:-1] + path[c:b - 1:-1] + path[f:]  # 2-opt

    return new_path

"""
    LeeSin uses 3-OPT local search algorithm
"""
class LeeSin(Agent):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super().__init__(x, y, file_name, max_elapsed_time)

    def get_agent_path(self, coin_distance):
        end_time = time.time() + int(self.max_elapsed_time - 1)

        path = [i for i in range(0, len(coin_distance))]
        cost = calculate_cost(path, coin_distance)

        while time.time() < end_time:
            new_path = three_opt(path, True)

            new_cost = calculate_cost(new_path, coin_distance)

            if new_cost < cost:
                cost = new_cost
                path = list(new_path)

        path = path + [0]
        return path


"""
    Kaisa uses Dynamic Programming to solve "brute-force" in O(N * 2 ^ N) instead of O(N!)
"""
class Kaisa(Agent):
    def __init__(self, x, y, file_name, max_elapsed_time):
        super().__init__(x, y, file_name, max_elapsed_time)

    def get_agent_path(self, coin_distance):
        num_coins = len(coin_distance)

        max_mask = 1 << num_coins
        # dp[node][mask] represents that we have collected coins represented by the mask and the last coin that was collected was node
        dp = [[sys.maxsize] * max_mask for _ in range(num_coins)]
        # prev is used to reconstruct the path
        prev = [[None] * max_mask for _ in range(num_coins)]

        for mask in range(1, max_mask):
            for node in range(0, num_coins):
                dp[node][mask] = sys.maxsize
                prev[node][mask] = None

                if (mask & (1 << node)) == 0:
                    continue

                if mask == (1 << node):
                    if node == 0:
                        dp[node][mask] = 0
                        prev[node][mask] = None
                else:
                    for previous_node in range(0, num_coins):
                        if previous_node == node:
                            continue

                        if (mask & (1 << previous_node)) == 0:
                            continue

                        cost = coin_distance[previous_node][node] + dp[previous_node][mask ^ (1 << node)]
                        if cost < dp[node][mask]:
                            dp[node][mask] = cost
                            prev[node][mask] = previous_node

        min_cost = sys.maxsize
        min_cost_index = -1
        for node in range(1, num_coins):
            cost = dp[node][max_mask - 1] + coin_distance[node][0]
            if cost < min_cost:
                min_cost = cost
                min_cost_index = node

        best_path = [0, min_cost_index]
        current_node = min_cost_index
        current_mask = max_mask - 1
        for _ in range(0, num_coins - 1):
            next_node = prev[current_node][current_mask]
            current_mask = current_mask ^ (1 << current_node)
            current_node = next_node
            best_path.append(current_node)

        # technically no need to reverse since the graph is symmetric
        return best_path[::-1]
