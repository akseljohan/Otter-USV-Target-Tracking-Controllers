import shutil
import time

import numpy as np
from matplotlib import pyplot as plt
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, VecNormalize

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

models_dir = "../trash/models/PPO/"
file_name = "best_model"

# mod3el_path =f"AI_controller/logs/PPO_end_when_reaching_target_true/1000_150_5_True_0_1_1_1_0_1/2023-03-25 19-43-22/best_model.zip"
# f"AI_controller/logs/PPO_end_when_reaching_target_true/1000_150_5_True_0_1_1_1_0_1/2023-03-25 19-43-25/best_model.zip"
model_path =  "AI_controller/logs/PPO_moving_target_normalized/1000_30_1_True_0_2_1_1_0_1/2023-04-09 17-13-42/best_model.zip"
    #f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.01_0.1_0_0_1/2023-04-16 17-35-34/models/PPO_moving_target_normalized__78900000_steps.zip"
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.1_0.1_0_1_1_3e-06/2023-04-17 17-54-54/models/PPO_moving_target_normalized__29500000_steps.zip"
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.01_0.1_0_1_1/2023-04-16 22-31-27/models/PPO_moving_target_normalized__67700000_steps.zip"
#
#

#these have plotted results:
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.1_0.1_0_1_1_3e-06/2023-04-17 17-54-54/models/PPO_moving_target_normalized__29500000_steps.zip"
    #f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.01_0.1_0_1_1/2023-04-16 22-31-27/models/PPO_moving_target_normalized__67700000_steps.zip"
#these have not plotted results
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.01_0.1_0_0_0/2023-04-15 15-17-49/models/PPO_moving_target_normalized__69000000_steps.zip"
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.01_0.1_0_0_1/2023-04-16 17-35-34/models/PPO_moving_target_normalized__64200000_steps.zip"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.01_0.1_0_0_0/2023-04-15 15-17-49/models/PPO_moving_target_normalized__69000000_steps.zip"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.1_0.1_0_1_1_3e-06/2023-04-17 17-54-54/models/PPO_moving_target_normalized__17800000_steps.zip"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.1_0.1_0_0_0/2023-04-13 16-43-39/models/PPO_moving_target_normalized__8000000_steps.zip"
# "AI_controller\\logs\\PPO_moving_target_normalized\\1000_50_1_True_1_0_0.1_0.1_0_0_0\\2023-04-11 22-09-23\\models\\PPO_moving_target_normalized__3500000_steps.zip"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_50_5_True_1_0_0.1_0.1_0_0_0/2023-04-11 12-21-40/models/PPO_moving_target_normalized__2000000_steps.zip"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_30_5_False_1_0_0_0.1_0_0_0/2023-04-11 09-43-59/models/PPO_moving_target_normalized__2000000_steps.zip"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_30_5_False_0_1_0_0.1_0_1_1/2023-04-11 08-26-41/models/PPO_moving_target_normalized__1000000_steps.zip"

vec_normalized_path = f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.01_0.1_0_0_1/2023-04-16 17-35-34/models/PPO_moving_target_normalized__vecnormalize_78900000_steps.pkl"
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.1_0.1_0_1_1_3e-06/2023-04-17 17-54-54/models/PPO_moving_target_normalized__vecnormalize_29500000_steps.pkl"
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.01_0.1_0_1_1/2023-04-16 22-31-27/models/PPO_moving_target_normalized__vecnormalize_67700000_steps.pkl"
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.1_0.1_0_1_1_3e-06/2023-04-17 17-54-54/models/PPO_moving_target_normalized__vecnormalize_29500000_steps.pkl"
#
#the models below has ploted results
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.1_0.1_0_1_1_3e-06/2023-04-17 17-54-54/models/PPO_moving_target_normalized__vecnormalize_29500000_steps.pkl"
    #

os.path.dirname(vec_normalized_path)
#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.01_0.1_0_0_0/2023-04-15 15-17-49/models/PPO_moving_target_normalized__vecnormalize_69000000_steps.pkl"

#f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.01_0.1_0_0_1/2023-04-16 17-35-34/models/PPO_moving_target_normalized__vecnormalize_64200000_steps.pkl"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.01_0.1_0_0_0/2023-04-15 15-17-49/models/PPO_moving_target_normalized__vecnormalize_69000000_steps.pkl"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_0_1_0.1_0.1_0_1_1_3e-06/2023-04-17 17-54-54/models/PPO_moving_target_normalized__vecnormalize_17800000_steps.pkl"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_50_1_True_1_0_0.1_0.1_0_0_0/2023-04-13 16-43-39/models/PPO_moving_target_normalized__vecnormalize_8000000_steps.pkl"
# f"AI_controller\\logs\\PPO_moving_target_normalized\\1000_50_1_True_1_0_0.1_0.1_0_0_0\\2023-04-11 22-09-23\\models\\PPO_moving_target_normalized__vecnormalize_3500000_steps.pkl"
# This is good:
# f"AI_controller/logs/PPO_moving_target_normalized/1000_50_5_True_1_0_0.1_0.1_0_0_0/2023-04-11 12-21-40/models/PPO_moving_target_normalized__vecnormalize_2000000_steps.pkl"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_30_5_False_1_0_0_0.1_0_0_0/2023-04-11 09-43-59/models/PPO_moving_target_normalized__vecnormalize_2000000_steps.pkl"
# f"AI_controller/logs/PPO_moving_target_normalized/1000_30_5_False_0_1_0_0.1_0_1_1/2023-04-11 08-26-41/models/PPO_moving_target_normalized__vecnormalize_1000000_steps.pkl"


vehicle = otter('TargetTrackingAI', 0.0, 0.0, 0, 000.0)

env = TargetTrackingEnv()
# env.render_mode = 'human'
env.vehicle = vehicle
env.vehicle.target = config['env']['fixed_target']
env = make_vec_env(lambda: env, n_envs=1)
n_stack = config['n_stacked_frames']
env = VecFrameStack(env, n_stack=n_stack)
env = VecNormalize.load(load_path=vec_normalized_path, venv=env)  # load normalization paramaters from training
sampleTime = config['sample_time']
# env.reset()


# loading the model
model = PPO.load(model_path, env=env)
print(model.policy)
print(f"model.observation.shape: {model.observation_space.shape}")
print(f"model.n_steps: {model.n_steps}")
print(f"model.action_noise: {model.action_noise}")
print(f"model.num_timesteps: {model.num_timesteps}")
episodes = 1
time_compute_controller_signals = []
for ep in range(episodes):
    print("New episode")
    # env.set_attr(attr_name='cycle_counter', value= 0)
    obs = env.reset()
    # env.set_attr(attr_name='cycle_counter', value=0, indices=0)
    # print(env.obs_rms.var)
    #print(obs)
    done = False
    trajectory = np.array([[0], [0], [0]])
    # print(trajectory)
    rewards = 0
    rew_tot = 0
    target = vehicle.target
    rew_trajectory = []

    while not done:
        trajectory = env.get_attr('trajectory')[
            0]  # when using vectorized environments it is reset when done, therfore we must fetch the trajectory before the last run

        action, _states = model.predict(obs, deterministic=True)
        start_time = time.time()
        obs, rewards, done, info = env.step(action)
        end_time = time.time()  # end timing how long it takes to compute controller signals
        time_compute_controller_signals.append(end_time - start_time)  # log

        rew_trajectory.append(rewards)

        rew_tot += rewards
    dir_name = os.path.dirname(os.path.dirname(vec_normalized_path))
    hyper_params = os.path.basename(os.path.dirname(dir_name))
    result_dir = os.path.join('results', vehicle.controlMode,
                              f"{hyper_params} {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")  # create a log dir for the results
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    #copy the config file
    shutil.copy(src=f'{os.path.dirname(os.path.dirname(vec_normalized_path))}\config.yml', dst=f"{result_dir}/copy_config.yml")  # copy the config file for documentation

    print(f"rew_tot: {rew_tot}")
    vehicle = env.get_attr('vehicle')[0]
    plt.plot(list(range(0, len(rew_trajectory))), rew_trajectory, label='reward history')
    plt.title(f"run: {ep}")
    plt.show()
    target_trajectory = trajectory[[6, 7], :]
    # print results

    pos_3dof = trajectory[[0, 1, 2, 3, 4, 5], :]
    utils.plot_trajectory(trajectory=pos_3dof, target=target_trajectory,
                          file_path=f'{result_dir}/AI_{hyper_params}_NED.pdf')

    # plot error plot
    utils.plot_error(trajectory[[0, 1], :].T, pos_target=target_trajectory.T, path=f'{result_dir}/AI_{hyper_params}_error.pdf',
                     sample_time=sampleTime)

    # plot the controller signals
    """utils.plot_controls(u_control=simData[:, [12, 13]],
                        u_actual=simData[:, [14, 15]],
                        sample_time=sampleTime,
                        file_path=f'{result_dir}/controls.pdf')"""
    print(trajectory.shape)
    utils.plot_veloceties(vel_matrix_3DOF=trajectory[[3, 4, 5], :],
                          #action_matrix=trajectory[[14, 15], :],
                          sample_time=sampleTime,
                          file_path=f'{result_dir}/AI_{hyper_params}_velocities.pdf')

    utils.plot_solv_time(solv_time_data=time_compute_controller_signals, sample_time=sampleTime, file_path=f'{result_dir}/AI_{hyper_params}_compute_time.pdf')

    utils.plot_control_forces(tau=trajectory[[14, 15], :].T, sample_time=sampleTime, file_path=f'{result_dir}/AI_{hyper_params}_controls.pdf')
    plt.show()