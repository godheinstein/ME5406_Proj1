# ME 5406 Deep Learning for Robotics Project1

# Problem Statement
Consider a frozen lake with (four) holes covered by patches of very thin ice. Suppose that a robot is to glide on the frozen surface from one location(i.e., the top left corner) to another (bottom right corner) in order to pick up a frisbee.
 
# 3 Model Free RL Algorithms:
1. Q learning
2. SARSA
3. First-visit Monte Carlo control

# File Contents
```shell
MES5406_PROJ1/
├── images/                       # Contains visual assets (e.g., robot, goal, hole, and lake images)
├── results/                      # Stores output files (e.g., plot results and Q tables)
├── frozenlake_env.py             # Custom FrozenLake environment implementation
├── monte_carlo_4x4.ipynb         # Monte Carlo algorithm implementation for 4x4 grid
├── monte_carlo_10x10.ipynb       # Monte Carlo algorithm implementation for 10x10 grid
├── sarsa_4x4.ipynb               # SARSA algorithm implementation for 4x4 grid
├── sarsa_10x10.ipynb             # SARSA algorithm implementation for 10x10 grid
├── Q learning_4x4.ipynb          # Q-learning algorithm implementation for 4x4 grid
├── Q learning_10x10.ipynb        # Q-learning algorithm implementation for 10x10 grid
├── requirements.txt              # Python dependencies for the project
└──  README.md                    # Project documentation (this file)
```

# Environment Setup 
Ensure you are in the folder directory 
```shell
conda create -n ME5406_env python==3.6
conda activate ME5406_env
conda install --file requirements.txt
```

# Requirements
Use `conda install --file requirements.txt` to install the following requirements:
- matplotlib
- numpy
- pandas
- pillow
- ipykernel

Should any of the requirements fail to install, use `pip install matplotlib numpy pandas pillow ipykernel`

(you can refer to requirements.txt)

# How to run 
1. Select the Jupyter notebook to run the specific RL algorithm and grid size. 
For example, if I want to run SARSA with 4x4, I will select the sarsa_4x4.ipynb notebook. 
2. After selecting the notebook, the results are already displayed. 
3. If you want to run the notebook, click on clear all outputs and click run all. 



