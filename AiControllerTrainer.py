import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

from AI_controller.gym.env.MarineVehicleTargetTrackingEnv import TargetTrackingEnv
from otter.otter import otter
from stable_baselines3 import PPO
import os
from config import config
from datetime import datetime
from stable_baselines3.common.logger import configure
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, \
    VecFrameStack, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper

# fetch config
th.autograd.set_detect_anomaly(True)
continue_training = config['continue_training']
debug = config['env']['debug']
vehicle = otter(0, 0, 0, 0, 0)
vehicle.target = config['env']['fixed_target']
n_stack = config['n_stacked_frames']

model_dir = os.path.join('AI_controller', 'models', config['model'],
                         f"{config['env']['time_out']}_"
                         f"{config['env']['target_spawn_region']}_"
                         f"{config['env']['target_confident_region']}_"
                         f"{config['env']['random_target']}_"
                         f"{config['env']['c1']}_"
                         f"{config['env']['c2']}_"
                         f"{config['env']['c3']}_"
                         f"{config['env']['c4']}_"
                         f"{config['env']['c5']}_"
                         f"{config['env']['c6']}",
                         datetime.now().strftime('%Y-%m-%d %H-%M-%S'))

log_dir = os.path.join('AI_controller', 'logs', config['model'],
                       f"{config['env']['time_out']}_"
                       f"{config['env']['target_spawn_region']}_"
                       f"{config['env']['target_confident_region']}_"
                       f"{config['env']['random_target']}_"
                       f"{config['env']['c1']}_"
                       f"{config['env']['c2']}_"
                       f"{config['env']['c3']}_"
                       f"{config['env']['c4']}_"
                       f"{config['env']['c5']}_"
                       f"{config['env']['c6']}",
                       datetime.now().strftime('%Y-%m-%d %H-%M-%S'))

model_prefix = config['model'] + '_'

if debug:
    model_dir = os.path.join('AI_controller', 'models', 'debug', config['model'],
                             datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
    log_dir = os.path.join('AI_controller', 'logs', 'debug', config['model'])
    model_prefix = config['model'] + '_'

if not os.path.exists('./' + model_dir):
    os.makedirs(model_dir)

env = TargetTrackingEnv()
env.vehicle = vehicle
env = make_vec_env(lambda: env, n_envs=1)
env = VecFrameStack(env, n_stack=n_stack)
env.reset()


def lrsched(alpha0):
    def reallr(progress):
        return alpha0 * np.exp(-(1 - progress))

    return reallr


policy_kwargs = None

if continue_training:
    print("Loading trained model")
    # finding existing model
    log_dir = 'AI_controller/logs/PPO/120_100_1_True_0_100_1_1_100_5/2023-03-08 17-36-18/'
    model_dir = 'AI_controller/logs/PPO/120_100_1_True_0_100_1_1_100_5/2023-03-08 17-36-18/'
    continue_from_model = 'best_model'
    model = PPO.load(f"{model_dir}/{continue_from_model}", env=env)
    reset_num_timesteps = False

else:
    # defining the new model
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=[32,32], vf=[32,32]))
    reset_num_timesteps = True
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, learning_rate=lrsched(0.0003),
                policy_kwargs=policy_kwargs)

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
# Separate evaluation env
eval_env = TargetTrackingEnv()
eval_env.vehicle = vehicle  # define vehicle for environment (should be passed when initializing)
eval_env = make_vec_env(lambda: eval_env, n_envs=1)
eval_env = VecFrameStack(eval_env, n_stack=n_stack)


# Stop training if there is no improvement after more than 5 evaluations
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=4, min_evals=5, verbose=1)

eval_callback = EvalCallback(eval_env, eval_freq=10_000,
                             #callback_after_eval=stop_train_callback,
                             n_eval_episodes=50,
                             best_model_save_path=log_dir,
                             deterministic= True,
                             render= False,
                             verbose=1, )

print(model.policy)

model.learn(total_timesteps=config['total_timesteps'],
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=f"PPO",
            callback=[eval_callback])  #

# test model

obs = env.reset()
for i in range(config['env']['time_out']):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()

env.close()
