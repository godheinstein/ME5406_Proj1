import os
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from PIL import Image, ImageDraw


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

        images_folder = os.path.join(os.path.dirname(__file__), 'images')
        self.robot_img = Image.open(os.path.join(images_folder, 'robot.png'))
        self.goal_img = Image.open(os.path.join(images_folder, 'frisbee.png'))
        self.hole_img = Image.open(os.path.join(images_folder, 'hole.png'))
        self.bg_img = Image.open(os.path.join(images_folder, 'lake.png'))

        self.cell_size = 50  
        self.window_size = self.grid_size * self.cell_size  

        self.robot_img = self.robot_img.resize((self.cell_size, self.cell_size))
        self.hole_img = self.hole_img.resize((self.cell_size, self.cell_size))
        self.goal_img = self.goal_img.resize((self.cell_size, self.cell_size))
        self.bg_img = self.bg_img.resize((self.window_size, self.window_size))
        
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
    # created a breath-first search to check if there is a path from start to goal
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

    # Define the movement for each action
    def step(self, action): 
        movement = {
            0: (-1, 0),  # UP
            1: (1, 0),   # DOWN
            2: (0, 1),   # RIGHT
            3: (0, -1)   # LEFT
        }
        
        # Calculate the new state
        new_state = (self.state[0] + movement[action][0], self.state[1] + movement[action][1])
        
        # Check if the new state is out of bounds
        if new_state[0] < 0 or new_state[0] >= self.grid_size or new_state[1] < 0 or new_state[1] >= self.grid_size:
            new_state = self.state  # Stay in the same state if out of bounds
        
        # Check if the new state is a hole or the goal
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
        self.path_dir[self.i] = new_state
        self.i += 1
        state_index = self.position_transition(*new_state)

        return state_index, reward, done


    def render(self):
        # Create a blank canvas
        canvas = Image.new('RGB', (self.window_size, self.window_size))
        
        # Draw the background
        canvas.paste(self.bg_img, (0, 0))
        
        # Draw the holes
        for hole in self.holes:
            x, y = hole
            canvas.paste(self.hole_img, (y * self.cell_size, x * self.cell_size))
        
        # Draw the goal
        goal_x, goal_y = self.goal
        canvas.paste(self.goal_img, (goal_y * self.cell_size, goal_x * self.cell_size))
        
        # Draw the robot
        robot_x, robot_y = self.state
        canvas.paste(self.robot_img, (robot_y * self.cell_size, robot_x * self.cell_size))
        
        # Display the canvas
        plt.imshow(canvas)
        plt.axis('off')
        plt.show()
 
    def final(self):
        print('shortest path:', self.shortest)
        print('longest path:', self.longest)
        print('the shortest route is shown in red')

        canvas = Image.new('RGB', (self.window_size, self.window_size))
        canvas.paste(self.bg_img, (0, 0))
        
        # Draw the holes
        for hole in self.holes:
            x, y = hole
            canvas.paste(self.hole_img, (y * self.cell_size, x * self.cell_size))
        
        # Draw the goal
        goal_x, goal_y = self.goal
        canvas.paste(self.goal_img, (goal_y * self.cell_size, goal_x * self.cell_size))
        
        # Draw the robot
        robot_x, robot_y = self.state
        canvas.paste(self.robot_img, (robot_y * self.cell_size, robot_x * self.cell_size))
        
        # Draw the shortest path
        draw = ImageDraw.Draw(canvas)
        for pos in self.path_store_dir.values():
            x = pos[1] * self.cell_size + self.cell_size // 2
            y = pos[0] * self.cell_size + self.cell_size // 2
            draw.ellipse((x-5, y-5, x+5, y+5), fill=(255, 0, 0))
        
        # Display the canvas
        plt.imshow(canvas)
        plt.axis('off')
        plt.show()

