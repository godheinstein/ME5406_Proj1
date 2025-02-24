import random
import numpy as np
from collections import defaultdict 

class MonteCarloControl:
    def __init__(self, env, gamma=0.99, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.returns = defaultdict(list)
        self.episode_rewards = []

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.env.actions.keys()))  # Explore
        else:
            return max(self.env.actions.keys(), key=lambda a: self.Q[state][a])  # Exploit

    def train(self, episodes):
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            episode_history = []
            episode_rewards = 0
            done = False

            # Generate episode
            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                episode_history.append((state, action, reward))
                episode_rewards += reward
                state = next_state
            rewards.append(episode_rewards)
        
            # Update Q-values using first-visit MC
            G = 0
            for t in range(len(episode_history) - 1, -1, -1):
                state, action, reward = episode_history[t]
                G = self.gamma * G + reward
                if (state, action) not in [(x[0], x[1]) for x in episode_history[:t]]:
                    self.returns[(state, action)].append(G)
                    self.Q[state][action] = np.mean(self.returns[(state, action)])

            # Record episode reward
            self.episode_rewards.append(episode_rewards)

        return self.Q, self.episode_rewards

class SARSA:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.episode_rewards = []

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.env.actions.keys()))  # Explore
        else:
            return max(self.env.actions.keys(), key=lambda a: self.Q[state][a])  # Exploit

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            action = self.choose_action(state)
            done = False
            total_reward = 0

            while not done:
                next_state, reward, done = self.env.step(action)
                next_action = self.choose_action(next_state)
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * self.Q[next_state][next_action] - 
                    self.Q[state][action]
                )
                total_reward += reward
                state = next_state
                action = next_action

            # Record episode reward
            self.episode_rewards.append(total_reward)

        return self.Q, self.episode_rewards


class QLearning:
    def __init__(self, env, gamma=0.99, alpha=0.1, epsilon=0.1):
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.episode_rewards = []

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(list(self.env.actions.keys()))  # Explore
        else:
            return max(self.env.actions.keys(), key=lambda a: self.Q[state][a])  # Exploit

    def train(self, episodes):
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)
                if not self.Q[next_state]:
                    self.Q[next_state] = defaultdict(float)
                if len(self.Q[next_state].values()) == 0:
                    self.Q[next_state] = {a: 0 for a in self.env.actions.keys()}
                self.Q[state][action] += self.alpha * (
                    reward + self.gamma * max(self.Q[next_state].values()) - 
                    self.Q[state][action]
                )
                total_reward += reward
                state = next_state

            # Record episode reward
            self.episode_rewards.append(total_reward)

        return self.Q, self.episode_rewards