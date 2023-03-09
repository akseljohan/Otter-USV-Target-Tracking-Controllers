import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack

import utils
from AI_controller.gym.env.MarineVehicleTargetTrackingEnv import TargetTrackingEnv
from otter.otter import otter
from stable_baselines3 import PPO
import os
from config import config
from datetime import datetime
from stable_baselines3.common.logger import configure
import torch as th
debug = False

models_dir = "models/PPO/"
file_name = "best_model"

model_path = f"AI_controller/logs/PPO/120_100_1_True_0_100_1_1_100_5/2023-03-08 17-36-18/best_model.zip"

vehicle = otter(0, 0, 0, 0, 0)

env = TargetTrackingEnv()
#env.render_mode = 'human'
env.vehicle = vehicle
env.vehicle.target = config['env']['fixed_target']
env = make_vec_env(lambda: env, n_envs=1)
n_stack = config['n_stacked_frames']
env = VecFrameStack(env, n_stack=n_stack)
#env.reset()




# loading the model
model = PPO.load(model_path, env=env)
print(model.policy)
print(model.observation_space)
episodes = 20

for ep in range(episodes):
    obs = env.reset()
    done = False
    trajectory = np.array([[0],[0], [0]])
    #print(trajectory)
    rewards = 0

    target = vehicle.target
    while not done:
        action, _states = model.predict(obs, deterministic=True )
        #print(f"model prediction returns: {model.predict(obs)}")
        #print(f"action: {action}" )
        #print(f"states: {_states}")
        #print(vehicle.target)
        obs, rewards, done, info = env.step(action)
        rewards +=rewards
        print(np.reshape(obs, (4,12)))
        print(f"rewards: {rewards}"
              f"done: {done}")
        trajectory = np.append(trajectory, np.reshape(obs[:1, -12:-9][0], (3,1)), axis = 1)
        np.set_printoptions(suppress= True, linewidth= 30000)
    print(trajectory)
    vehicle = env.get_attr('vehicle')[0]
    print(obs)
    print(obs[:,-5:-3])
    utils.plot_trajectory(trajectory = trajectory[:, :-1], target= target )
