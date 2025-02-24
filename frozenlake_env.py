import os
import random
import numpy as np
import pygame
from collections import defaultdict 
from collections import deque

class FrozenLakeEnv:
    def __init__(self, grid_size=4, hole_fraction=0.25, use_default_map = False):
        self.grid_size = grid_size
        self.hole_fraction = hole_fraction
        self.use_default_map = use_default_map  
        self.actions = {0: 'UP', 1: 'DOWN', 2: 'RIGHT', 3: 'LEFT'}
        self.map = self.generate_map()
        self.state = self.find_start_position()
        self.goal = self.find_goal_position()
        self.holes = self.find_holes()
        self.window_size = 400
        self.cell_size = self.window_size // grid_size
        pygame.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption('Frozen Lake')

        images_folder = os.path.join(os.path.dirname(__file__), 'images')
        self.images = images_folder 
        self.robot_img = pygame.image.load(os.path.join(images_folder, 'robot.png'))
        self.goal_img = pygame.image.load(os.path.join(images_folder, 'frisbee.png'))
        self.hole_img = pygame.image.load(os.path.join(images_folder, 'hole.png'))
        self.bg_img = pygame.image.load(os.path.join(images_folder, 'lake.png'))

        self.robot_img = pygame.transform.scale(self.robot_img, (self.cell_size, self.cell_size))
        self.hole_img = pygame.transform.scale(self.hole_img, (self.cell_size, self.cell_size))
        self.goal_img = pygame.transform.scale(self.goal_img, (self.cell_size, self.cell_size))
        self.bg_img = pygame.transform.scale(self.bg_img, (self.window_size, self.window_size))

        self.map = self.generate_map()
        self.state = self.find_start_position()
        self.goal = self.find_goal_position()
        self.holes = self.find_holes()
        
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
            return self.generate_random_map()
        
    def generate_random_map(self):
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

            # Check if there's a valid path from start to goal
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

            for action in self.actions.values():
                nr, nc = current[0] + action[0], current[1] + action[1]
                if 0 <= nr < self.grid_size and 0 <= nc < self.grid_size:
                    if map[nr, nc] != 1:  
                        queue.append((nr, nc))

        return False        

    def find_start_position(self): 
        return tuple(np.argwhere(self.map == 2)[0])
    
    def find_goal_position(self):
        return tuple(np.argwhere(self.map == 3)[0])

    def find_holes(self):
        return set(tuple(hole) for hole in np.argwhere(self.map == 1))

    def reset(self):
        self.state = self.find_start_position()
        return self.state

    def step(self, action):
        move = self.actions.get(action, (0, 0))
        new_state = (max(0, min(self.grid_size - 1, self.state[0] + move[0])),
                     max(0, min(self.grid_size - 1, self.state[1] + move[1])))
        if new_state in self.holes:
            return new_state, -1, True
        elif new_state == self.goal:
            return new_state, 1, True
        else:
            self.state = new_state
            return new_state, 0, False

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


