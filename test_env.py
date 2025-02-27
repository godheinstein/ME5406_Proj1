import time
from frozenlake_env import FrozenLakeEnv

def test_frozen_lake_env():
    env = FrozenLakeEnv(grid_size=4, hole_fraction=0.25, use_default_map=True)
    
    # Reset the environment to the initial state
    state = env.state
    print(f"Initial state: {state}")
    
    # Define a sequence of actions to take
    actions = [2, 2, 1, 1, 1, 2, 2]  # Example actions: RIGHT, RIGHT, DOWN, DOWN, DOWN, RIGHT, RIGHT
    
    for action in actions:
        new_state, reward, done = env.step(action)
        print(f"Action: {env.actions[action]}, New state: {new_state}, Reward: {reward}, Done: {done}")
        
        # Render the environment
        env.render()
        
        if done:
            break
        
        # Add a small delay to see the rendering
        time.sleep(1)
    
    # Print final results
    env.final()

if __name__ == "__main__":
    test_frozen_lake_env()