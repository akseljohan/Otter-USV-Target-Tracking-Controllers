import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, StopTrainingOnNoModelImprovement, EvalCallback
from stable_baselines3.common.env_util import make_vec_env

import utils
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

# import modifiers
#np.seterr(all='raise')
def make_my_env():
    temp_env = TargetTrackingEnv()
    vehicle = otter(0, 0, 0, 0, 0)
    temp_env.vehicle = vehicle
    return temp_env
# fetch config
th.autograd.set_detect_anomaly(True)
continue_training = config['continue_training']
debug = config['env']['debug']
vehicle = otter(0, 0, 0, 0, 0)
n_stack = config['n_stacked_frames']

architecture = config['architecture']
starting_learning_rate = config['starting_learning_rate']


log_dir = os.path.join('AI_controller', 'trash/logs', config['model'],
                       f"{config['env']['time_out']}_"
                       f"{config['env']['target_spawn_region']}_"
                       f"{config['env']['target_confident_region']}_"
                       f"{config['env']['random_target']}_"
                       f"{config['env']['c1']}_"
                       f"{config['env']['c2']}_"
                       f"{config['env']['c3']}_"
                       f"{config['env']['c4']}_"
                       f"{config['env']['c5']}_"
                       f"{config['env']['c6']}_"
                       f"{config['env']['c7']}_"
                       f"{config['starting_learning_rate']}",
                       datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
model_dir  = os.path.join(log_dir, 'model')

model_prefix = config['model'] + '_'

if debug:
    model_dir = os.path.join('AI_controller', 'models', 'debug', config['model'],
                             datetime.now().strftime('%Y-%m-%d %H-%M-%S'))
    log_dir = os.path.join('AI_controller', 'trash/logs', 'debug', config['model'])
    model_prefix = config['model'] + '_'

if not os.path.exists('./' + log_dir):
    os.makedirs(log_dir)



#create training environment
env = TargetTrackingEnv()
env.vehicle = vehicle
env = make_vec_env(lambda: make_my_env(), n_envs=config['n_env'])
env = VecFrameStack(env, n_stack=n_stack)
env = VecNormalize(venv=env)
env.reset()

# Separate evaluation env
eval_env = TargetTrackingEnv()
eval_env.vehicle = otter(0, 0, 0, 0, 0)  # define vehicle for environment (should be passed when initializing)
eval_env = make_vec_env(lambda: eval_env, n_envs=1)
eval_env = VecFrameStack(eval_env, n_stack=n_stack)
eval_env = VecNormalize(venv=eval_env)
eval_env.reset()

def lrsched(alpha0):
    def reallr(progress):
        return alpha0 * np.exp(-(1 - progress))

    return reallr


policy_kwargs = None

if continue_training:
    print("Loading trained model_and continuing training")
    # finding existing model
    tmp_log_dir = config['log_dir']
    #model_dir = 'AI_controller/logs/PPO/120_100_1_True_0_100_1_1_100_5/2023-03-08 17-36-18/'
    continue_from_model = 'best_model'
    model = PPO.load(f"{tmp_log_dir}/{continue_from_model}", env=env)
    reset_num_timesteps = True

else:
    print("Starting new training session")
    policy_kwargs = dict(activation_fn=th.nn.ReLU,
                         net_arch=dict(pi=architecture, vf=architecture))
    reset_num_timesteps = True
    model = PPO("MlpPolicy", env=env, verbose=1, tensorboard_log=log_dir, learning_rate=lrsched(starting_learning_rate),
                policy_kwargs=policy_kwargs, device='cuda')

# Defining callbacks

checkpoint_callback = CheckpointCallback(
    save_freq= max(100_000 // config['n_env'], 1),
    save_path=log_dir+'/models',
    name_prefix=model_prefix,
    save_replay_buffer=True,
    save_vecnormalize=True,
    # verbose=1
    )
# Stop training if there is no improvement after more than 5 evaluations
stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=4, min_evals=5, verbose=1)

eval_callback = EvalCallback(eval_env, eval_freq=max(100_000 // config['n_env'], 1),
                             #callback_after_eval=stop_train_callback,
                             n_eval_episodes=10,
                             best_model_save_path=log_dir,
                             deterministic= True,
                             render= False,
                             verbose=1, )
utils.copy_config(log_dir)#copy the config file for documentation
print(model.policy)

model.learn(total_timesteps=config['total_timesteps'],
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=f"PPO_arch_{str(architecture)}",
            callback=[checkpoint_callback, eval_callback])  #

# save the normalization statistics (to be used in inference)
env.save(log_dir+f"\\norm.pickle")

# test model

obs = env.reset()
trajectory = []
target = vehicle.target
done = False
while not done:
    trajectory = env.get_attr('trajectory')[0] # we must call this first because the stacked env is reseting the environment when done.
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    #env.render()


utils.plot_trajectory(trajectory= trajectory, target=target)
env.close()
