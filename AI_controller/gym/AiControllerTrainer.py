import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from AI_controller.gym.env.MarineVehicleTargetTrackingEnv import TargetTrackingEnv
from otter.otter import otter
from stable_baselines3 import PPO
import os
from config import config
from datetime import datetime
from stable_baselines3.common.logger import configure
import torch as th

th.autograd.set_detect_anomaly(True)
debug = config['env']['debug']

vehicle = otter(0, 0, 0, 0, 0)
# vehicle.target = [10, 10]
model_dir = os.path.join('AI_controller', 'models', config['model'], datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
log_dir = os.path.join('AI_controller', 'logs', config['model'])
model_prefix = config['model'] + '_'

env = TargetTrackingEnv()

if debug:
    model_dir = os.path.join('AI_controller', 'models', 'debug', config['model'],
                             datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
    log_dir = os.path.join('AI_controller', 'logs', 'debug', config['model'])
    model_prefix = config['model'] + '_'

if not os.path.exists('./' + model_dir):
    os.makedirs(model_dir)

# env.render_mode = 'human'
env.vehicle = vehicle
# env = Monitor(env, log_dir, force=True)
# env = SnekEnv()
env.reset()

# Defining callbacks
# Save a checkpoint every 1000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path=model_dir,
    name_prefix=model_prefix,
    save_replay_buffer=True,
    save_vecnormalize=True,
    # verbose=1
)


def lrsched(alpha0):
    def reallr(progress):
        """lr = 0.003
        if progress < 0.85:
            lr = 0.0005
        if progress < 0.66:
            lr = 0.00025
        if progress < 0.33:
            lr = 0.0001
        return lr"""
        return alpha0*np.exp(-(1-progress))

    return reallr





# defining the model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=lrsched(0.0003))

configure()
# Set new logger
model.learn(total_timesteps=100_000_0, reset_num_timesteps=True, tb_log_name=f"PPO", callback=[checkpoint_callback])  #

# test model
"""vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()"""
