import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from frozenlake_env import FrozenLakeEnv

# set random seed for reproducibility
np.random.seed(25)
random.seed(25)

# define hyperparameters
GAMMA = 0.99  # discount factor
EPSILON = 0.1  # exploration rate
EPISODES = 2000  # no. of episodes
EPSILON_DECAY = 0.995  # epsilon decay rate
EPSILON_MIN = 0.01  # minimum epsilon
GRID_SIZE = 4  # grid size (4x4 or 10x10)
HOLE_FRACTION = 0.25  # fraction of holes
USE_DEFAULT_MAP = False  # use default map if available

# create environment
env = FrozenLakeEnv(grid_size=GRID_SIZE, hole_fraction=HOLE_FRACTION, use_default_map=USE_DEFAULT_MAP)

class MonteCarloControl: 
    def __init__(self, env, gamma=GAMMA, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_min=EPSILON_MIN):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.Q = defaultdict(lambda: np.zeros(len(env.actions)))
        self.returns = defaultdict(list)
        self.episode_rewards = []

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.env.actions.keys()))  # Explore
        else:
            return max(self.env.actions.keys(), key=lambda a: self.Q[state][a])  # Exploit

    def train(self, episodes):
        for episode in range(episodes):
            state = env.reset()
            episode_history = []
            done = False
            total_reward = 0

            # generate episode
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = env.step(action)
                episode_history.append((state, action, reward))
                total_reward += reward
                state = next_state

            # update Q-values using first-visit MC
            G = 0
            for t in range(len(episode_history) - 1, -1, -1):
                state, action, reward = episode_history[t]
                G = self.gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode_history[:t]]:
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])

            # decay epsilon
            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)  
            # record episode reward
            self.episode_rewards.append(total_reward)
        return self.Q, self.episode_rewards
    
    def visualize_policy(self, Q, env):
        policy = np.zeros((env.grid_size, env.grid_size), dtype=str)
        for state in Q:
            row, col = state // env.grid_size, state % env.grid_size
            action = np.argmax(Q[state])
            policy[row, col] = env.actions[action][0]  # Use first letter of action (U, D, R, L)
        print("Learned Policy:")
        print(policy)


    def test_policy(Q, env, episodes=10):
        total_rewards = 0
        for episode in range(episodes):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = np.argmax(Q[state])  # Greedy policy
                next_state, reward, done = env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards += episode_reward
            print(f"Episode {episode + 1}: Reward = {episode_reward}")
        print(f"Average Reward over {episodes} episodes: {total_rewards / episodes}")

    

    def visualize_value_function(Q, env):
        value_function = np.zeros((env.grid_size, env.grid_size))
        for state in Q:
            row, col = state // env.grid_size, state % env.grid_size
            value_function[row, col] = np.max(Q[state])
        print("Value Function:")
        print(value_function)

    def render_policy(Q, env):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = np.argmax(Q[state])  # Greedy policy
            next_state, reward, done = env.step(action)
            state = next_state
        env.final()  # Show the final path


    def save_q_table_text(self, filename):
        """
        Save the Q-table to a plain text file.
        """
        with open(filename, 'w') as f:
            for state, actions in self.Q.items():
                action_values = ' '.join(map(str, actions))
                f.write(f"{state} {action_values}\n")

    def load_q_table_text(self, filename):
        """
        Load the Q-table from a plain text file.
        """
        self.Q = defaultdict(lambda: np.zeros(len(self.env.actions)))
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                state = int(parts[0])
                actions = np.array([float(x) for x in parts[1:]])
                self.Q[state] = actions   


print("Training with Monte Carlo Control...")
mc_agent = MonteCarloControl(env)
Q_mc, mc_rewards = mc_agent.train(EPISODES)
print("Training complete!")

# save the Q-table
mc_agent.save_q_table_text('q_table_mc.txt')
print("Q-table saved to 'q_table_mc.txt'")

# load the Q-table (optional, if want to test without retraining)
mc_agent.load_q_table_text('q_table_mc.txt')
print("Q-table loaded from 'q_table_mc.txt'")

# visualize the policy 
mc_agent.visualize_policy(Q_mc, env)  
# test the policy
mc_agent.test_policy(Q_mc, env)
# visualize the value function
mc_agent.visualize_value_function(Q_mc, env)
# render the policy
mc_agent.render_policy(Q_mc, env)

# plot cumulative rewards
plt.figure(figsize=(12, 6))
plt.plot(np.cumsum(mc_rewards), label=f"Monte Carlo Control (Map Size: {GRID_SIZE}x{GRID_SIZE})")
plt.title("Cumulative Rewards Over Episodes")
plt.xlabel("Episodes")
plt.ylabel("Cumulative Reward")
plt.legend()
plt.grid()
plt.show()