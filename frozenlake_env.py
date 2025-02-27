import os
import random
import numpy as np
import pygame
from collections import defaultdict, deque


class FrozenLakeEnv:
    def __init__(self, grid_size=4, hole_fraction=0.25, use_default_map = False):
        self.grid_size = grid_size
        self.hole_fraction = hole_fraction
        self.use_default_map = use_default_map  
        self.state = (0,0)
        self.actions = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
        self.map = self.generate_map()
        self.goal = (grid_size -1, grid_size -1)
        self.holes = self.find_holes()
        self.path_dir = {}
        self.path_store_dir = {}
        self.i = 0 
        self.store_path = True
        self.longest = 0
        self.shortest = 0
        self.window_size = 400
        self.cell_size = self.window_size // grid_size
        pygame.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Frozen Lake')

        images_folder = os.path.join(os.path.dirname(__file__), 'images')
        self.robot_img = pygame.image.load(os.path.join(images_folder, 'robot.png'))
        self.goal_img = pygame.image.load(os.path.join(images_folder, 'frisbee.png'))
        self.hole_img = pygame.image.load(os.path.join(images_folder, 'hole.png'))
        self.bg_img = pygame.image.load(os.path.join(images_folder, 'lake.png'))

        self.robot_img = pygame.transform.scale(self.robot_img, (self.cell_size, self.cell_size))
        self.hole_img = pygame.transform.scale(self.hole_img, (self.cell_size, self.cell_size))
        self.goal_img = pygame.transform.scale(self.goal_img, (self.cell_size, self.cell_size))
        self.bg_img = pygame.transform.scale(self.bg_img, (self.window_size, self.window_size))
        
    def generate_map(self):
        if self.use_default_map:
            if self.grid_size == 4:
                return np.array([[2, 0, 0, 0],
                                [0, 1, 0, 1],
                                [0, 0, 0, 1],
                                [1, 0, 0, 3]])
            elif self.grid_size == 10:
                return np.array([
                        [2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
                        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0, 0, 3]
                    ])
            else: 
                raise ValueError('Default map only available for 4x4 and 10x10 grids.')
        else: 
            return self.generate_random_map_with_path()
        
    def generate_random_map_with_path(self):
        while True:
            map = np.zeros((self.grid_size, self.grid_size), dtype=int)
            map[0, 0] = 2 
            map[self.grid_size - 1, self.grid_size - 1] = 3  

            num_holes = int(self.grid_size**2 * self.hole_fraction)
            holes_added = 0
            while holes_added < num_holes:
                r, c = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                if map[r, c] == 0:
                    map[r, c] = 1
                    holes_added += 1

            # check if path valid
            if self.is_path_available(map):
                return map
    
    def is_path_available(self, map):
        start = (0,0)
        goal = (self.grid_size - 1, self.grid_size - 1)
        visited = set()
        queue = deque([start])

        while queue:
            current = queue.popleft()
            if current == goal:
                return True
            if current in visited:
                continue
            visited.add(current)

            moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in moves:
                nr, nc = current[0] + dr, current[1] + dc
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if map[nr, nc] != 1:  
                        queue.append((nr, nc))
        return False        

    # extract the positions of holes from the map
    def find_holes(self):
        
        holes = set()
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if self.map[i, j] == 1:
                    holes.add((i, j))
        return holes
    # convert (row, col) tuple to an integer index
    def position_transition(self, row, col):
        
        return row * self.grid_size + col
    # reset the environment and return the initial state
    def reset(self):
        
        self.state = (0, 0)
        self.path_dir = {}
        self.i = 0
        return self.position_transition(*self.state)

    def step(self, action):
        move = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, 1),   # RIGHT
            3: (0, -1)   # LEFT
        }.get(action, (0, 0))

        new_state = (
            max(0, min(self.grid_size - 1, self.state[0] + move[0])),
            max(0, min(self.grid_size - 1, self.state[1] + move[1]))
        )

        self.path_dir[self.i] = new_state
        self.i += 1

        if new_state in self.holes:
            reward = -1
            done = True
        elif new_state == self.goal:
            reward = 1
            done = True
            # Store the shortest path
            if self.store_path:
                self.path_store_dir = self.path_dir.copy()
                self.store_path = False
                self.shortest = len(self.path_dir)
                self.longest = len(self.path_dir)
            elif len(self.path_dir) < self.shortest:
                self.shortest = len(self.path_dir)
                self.path_store_dir = self.path_dir.copy()
            elif len(self.path_dir) > self.longest:
                self.longest = len(self.path_dir)
        else:
            reward = 0
            done = False

        self.state = new_state
        state_index = self.position_transition(*new_state)
        return state_index, reward, done


    def render(self):
        self.window.blit(self.bg_img, (0, 0))
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pos = (j * self.cell_size, i * self.cell_size)
                if (i, j) in self.holes:
                    self.window.blit(self.hole_img, pos)
                elif (i, j) == self.goal:
                    self.window.blit(self.goal_img, pos)
                elif (i, j) == self.state:
                    self.window.blit(self.robot_img, pos)
        pygame.display.update()

    def final(self): 
        print('shortest path:', self.shortest)
        print('longest path:', self.longest)
        print('the shortest route is shown in red')

        for j in range(len(self.path_store_dir)):
            pos = self.path_store_dir[j]
            x = pos[1] * self.cell_size + self.cell_size // 2
            y = pos[0] * self.cell_size + self.cell_size // 2
            pygame.draw.circle(self.window, (255, 0, 0), (x, y), 5)
        pygame.display.update()

