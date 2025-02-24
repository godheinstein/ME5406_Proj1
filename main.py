from frozenlake_env import FrozenLakeEnv
from parameters import *
from algorithms import MonteCarloControl, SARSA, QLearning
import matplotlib.pyplot as plt
import numpy as np
import argparse

def main(algorithm, map_size):
    # Create the environment
    env = FrozenLakeEnv(grid_size=map_size, hole_fraction=HOLE_FRACTION, use_default_map=USE_DEFAULT_MAP)

    # Train the selected algorithm
    if algorithm.lower() == "monte_carlo":
        print("Training with Monte Carlo Control...")
        agent = MonteCarloControl(env, gamma=GAMMA, epsilon=EPSILON)
        Q, rewards = agent.train(EPISODES)
    elif algorithm.lower() == "sarsa":
        print("Training with SARSA...")
        agent = SARSA(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON)
        Q, rewards = agent.train(EPISODES)
    elif algorithm.lower() == "q_learning":
        print("Training with Q-Learning...")
        agent = QLearning(env, gamma=GAMMA, alpha=ALPHA, epsilon=EPSILON)
        Q, rewards = agent.train(EPISODES)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose from 'monte_carlo', 'sarsa', or 'q_learning'.")

    print("Training complete!")

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot cumulative rewards
    plt.plot(np.cumsum(rewards), label=f"{algorithm.title()} (Map Size: {map_size}x{map_size})")

    plt.title("Cumulative Rewards Over Episodes")
    plt.xlabel("Episodes")
    plt.ylabel("Cumulative Reward")
    plt.legend()
    plt.grid()
    plt.show()

    # Plot moving average of rewards
    window_size = 100  # Moving average window
    plt.figure(figsize=(12, 6))

    plt.plot(np.convolve(rewards, np.ones(window_size)/window_size, mode='valid'), 
             label=f"{algorithm.title()} (Map Size: {map_size}x{map_size})")
    plt.title(f"Moving Average of Rewards (Window Size = {window_size})")
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run reinforcement learning algorithms on the Frozen Lake environment.")
    parser.add_argument("--algorithm", type=str, required=True, help="Algorithm to run: 'monte_carlo', 'sarsa', or 'q_learning'.")
    parser.add_argument("--map_size", type=int, required=True, help="Size of the map: 4 or 10.")
    args = parser.parse_args()

    # Run the main function with the selected algorithm and map size
    main(algorithm=args.algorithm, map_size=args.map_size)