import os
import random
import numpy as np
import pygame
from collections import defaultdict

class FrozenLakeEnv:
    def __init__(self, grid_size=4, hole_fraction=0.25):
        self.grid_size = grid_size
        self.hole_fraction = hole_fraction
        self.state = (0,0)
        self.goal = (grid_size-1, grid_size-1)
        self.
        self.holes = self.generate_holes()
        self.actions = {'L': (0, -1), 'R': (0, 1), 'U': (-1, 0), 'D': (1, 0)}
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

    def generate_map(self):
        if self.use_default_map:
            if self.grid_size == 4:
                return np.array([[0, 0, 0, 1],
                                [0, 1, 0, 1],
                                [0, 0, 0, 0],
                                [1, 0, 1, 0]])
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
        





    def generate_holes(self):
        num_holes = int(self.grid_size**2 * self.hole_fraction)
        holes = set()
        while len(holes) < num_holes:
            r, c = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            if (r, c) != self.goal and (r,c) != (0,0):
                holes.add((r, c))
        return holes
    
    def reset(self):
        self.state = (0, 0)
        return self.state
    
    def step(self, action):
        move = self.actions.get(action, (0,0))
        new_state = (max(0, min(self.grid_size-1, self.state[0] + move[0])),
                     max(0, min(self.grid_size-1, self.state[1] + move[1])))
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

env = FrozenLakeEnv(grid_size=10)
env.render()
state = env.reset()
done = False
while not done:
    pygame.time.delay(500)
    action = random.choice(list(env.actions.keys()))
    state, reward, done = env.step(action)
    env.render()

pygame.quit()

