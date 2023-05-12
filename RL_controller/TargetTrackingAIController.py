import numpy as np
from matplotlib import pyplot as plt

import utils
from stable_baselines3 import PPO
import os
from datetime import datetime
from stable_baselines3.common.logger import configure
import math
import yaml

# velocity
class TargetTrackingAIController:
    def __init__(self, model_name, config_name):
        """
        NOT WORKING YET! This class is not used to test teh AI controllers. The AIControllerTesting is used.

        A self-contained controller. Requier a config file in the sam dir as the model.
        Must also have access to the normalisation statistics for the observations.

        :param model_dir: path to dir where model and file is stored
        :param congif_name: path and to config file .yaml
        """

        self.model_name = model_name
        self.config = yaml.safe_load(open((config_name)))
        self.eta = None
        # self.target = config['env']['fixed_target']
        self.model = self.load_PPO_model(model_path=model_name)
        self.observation_shape = self.model.observation_space.shape
        self.n_stacked_frames = self.config['n_stacked_frames']
        self.obs_size = self.observation_shape[
                            0] / self.n_stacked_frames  # the amount of stacked observations for the agent to make its predictions o
        self.stacked_observations = self.__initiate_stacked_observations() #stacked observations for the model to make its predictions on
        print(f"observations.shape:{self.stacked_observations.shape}")
        self.trajectory = np.zeros((int(self.obs_size), 1))  # the trajectory of the observations
        self.sample_time = self.config['sample_time']
        self.sim_length = self.config['env']['sim_length']
        self.radius = self.config['env']['target_confident_region']
        self.time_out = self.config['env']['time_out']
        self.action = []

    def get_psi_d(self, eta, target):
        """Returns the angle between of the line from the vessel to the target  in NED"""
        return math.atan2(target[1] - eta[1], target[0] - eta[0])

    def get_smalles_sign_angle(self, x=None, y=None, angle=None):  # page 413. in Fossen 2021

        if angle is not None:
            return (angle + math.pi) % (2 * math.pi) - math.pi
        if x is not None and y is not None:
            a = (x - y) % (2 * np.pi)
            b = (y - x) % (2 * np.pi)
            return -a if a < b else b
        else:
            raise ValueError("you must define either x and y, or angle")

    def get_euclidean_distance(self, eta, target):
        return np.linalg.norm(np.array([eta[0], eta[1]]) - target)

    def get_speed_towards_target(self, trajectory, target):

        if trajectory.shape[1] > 1:
            pos_k2 = np.array([trajectory[0, -2], trajectory[1, -2]])
            pos_k1 = np.array([trajectory[0, -1], trajectory[1, -1]])
            return (np.linalg.norm(pos_k2 - target) - np.linalg.norm(
                pos_k1 - target)) / self.sim_length
        else:
            return 0

    def __initiate_stacked_observations(self):
        return np.zeros(shape=self.model.observation_space.shape)

    def stack_observations(self, observation_stack, new_observation):
        """
        :param new_observations: new observation elements (18,1) to be added to the stacked observations
        :return: the new stacked_observation matrix
        """
        #shifted_mat = np.roll(self.stacked_observations, -self.obs_size, axis=0) #shifts the

        tmp = np.hstack((observation_stack, new_observation))
        tmp = tmp[int(self.obs_size) :]
        if tmp.shape == self.model.observation_space.shape:
            #print(f"tmp.shape_after removal:{tmp.shape}")
            return tmp
        else:
            print(f"the new observation stack({tmp.shape}), does not match the model observation shape ({self.model.observation_space.shape})")
        return observation_stack

    def load_PPO_model(self, model_path):
        # loading the model
        try:
            return PPO.load(model_path, env=None)
        except Exception as e:
            print(e)

    def get_delta_pos_target(self, eta, target):
        delta_x = abs(eta[0] - target[0])
        delta_y = abs(eta[1] - target[1])
        return delta_x, delta_y

    def get_action_dot(self, sample_time):
        return (np.linalg.norm(self.trajectory[14:16, -1]) - np.linalg.norm(self.trajectory[14:16, -2])) / sample_time

    def set_up(self, initial_state, target):
        print("Setting up controller)")
        self.t = 0 # time is set to zero
        eta = initial_state[:3]  # internal 3DOF eta
        nu = initial_state[3:]  # internal 3DOF nu
        #print(self.model.predict(self.stacked_observations))
        # calculate and add observations to trajectory
        #self.action_trajectory = np.zeros((self.observation_space.shape[0],))
        # set observations
        observation = np.array([eta[0],  # x (N)
                                eta[1],  # y (E)
                                eta[2],  # psi (angle from north)
                                nu[0],  # surge vel.
                                nu[1],  # sway vel.
                                nu[2],  # yaw vel.
                                target[0],  # x, target
                                target[1],  # y, target
                                self.get_delta_pos_target(eta, target)[0],  # distance between vehicle and target in x
                                self.get_delta_pos_target(eta, target)[1],  # distance between vehicle and target in y
                                self.get_euclidean_distance(eta, target),  # euclidean distance vehicle and target
                                0,  # difference in the two previous actions
                                0,
                                self.get_smalles_sign_angle(angle=(self.get_psi_d(eta=eta, target=target) - eta[2])),
                                # the angle between the target and the heading of the vessle
                                0,  # action surge
                                0,  # action sway
                                0,  # time awareness,
                                self.radius  # aware of the region around the target
                                ], dtype=np.float32)
        self.trajectory = np.append(self.trajectory, np.array([observation]).T,
                                    axis=1)  # appends a trajectory of observations# initiate observations

    def get_controller_action(self, initial_state, target, sample_time, t ):
        #print(f" t: {t}, self.time_out: {self.time_out},t/self.sample_time :{t/self.sample_time},  t % self.time_out: {cc}")
        #print(t % self.time_out)
        if t ==0:
            self.set_up(initial_state=initial_state, target= target)
        elif (t/self.sample_time )% self.time_out ==0:
            print("Training time period reached")
            self.set_up(initial_state=initial_state, target=target)


        #if t> self.time_out:
        #    self.set_up()
        eta = initial_state[:3]  # internal 3DOF eta
        nu = initial_state[3:]  # internatl 3DOF nu
        #print(self.model.predict(self.stacked_observations))
        # calculate and add observations to trajectory

        #predict action absed on last timestep:
        norm_action = self.model.predict(observation=self.stacked_observations, deterministic=True)[0]
        norm_action = norm_action
        # set observations (just so that the action is tace
        observation = np.array([eta[0],  # x (N)
                                eta[1],  # y (E)
                                eta[2],  # psi (angle from north)
                                nu[0],  # surge vel.
                                nu[1],  # sway vel.
                                nu[2],  # yaw vel.
                                target[0],  # x, target
                                target[1],  # y, target
                                self.get_delta_pos_target(eta, target)[0],  # distance between vehicle and target in x
                                self.get_delta_pos_target(eta, target)[1],  # distance between vehicle and target in y
                                self.get_euclidean_distance(eta, target),  # euclidean distance vehicle and target
                                self.get_action_dot(sample_time=sample_time),  # difference in the two previous actions
                                self.get_speed_towards_target(trajectory=self.trajectory, target=target),
                                self.get_smalles_sign_angle(angle=(self.get_psi_d(eta=eta, target=target) - eta[2])),
                                # the angle between the target and the heading of the vessle
                                norm_action[0],  # action surge
                                norm_action[1],  # action sway
                                t / self.time_out,  # time awareness, #TODO set up a modulus for calculaing this since the simulation time can be larger than the training horizon?
                                1,#self.radius  # aware of the region around the target
                                ], dtype=np.float32)
        np.set_printoptions(linewidth= 2000,precision=3, formatter={'float': '{: 0.3f}'.format})
        #print(observation)
        self.trajectory = np.append(self.trajectory, np.array([observation]).T,
                                    axis=1)  # appends a trajectory of observations
        self.stacked_observations = self.stack_observations(self.stacked_observations, new_observation=observation)


        #print(f"norm_actions: {norm_action}")
        tau_X = utils.denormalize(self.config['constraints']['forces']['X']['lower'], self.config['constraints']['forces']['X']['upper'], norm_action[0])
        # print(f"tau_X: {tau_X}")
        tau_N = utils.denormalize(self.config['constraints']['forces']['N']['lower'], self.config['constraints']['forces']['N']['upper'], norm_action[1])
        #print(tau_X, tau_N)
        tau_X = tau_X
        tau_N = tau_N
        return tau_X, tau_N


if __name__ == '__main__':
    # loading the model
    from otter.otter import otter
    #Tests:
    model_name = f"AI_controller/logs/PPO_end_when_reaching_target_true/500_150_5_True_0_1_1_1_0_1/2023-03-28 09-17-53/best_model.zip"
    config_name = f"AI_controller/logs/PPO_end_when_reaching_target_true/500_150_5_True_0_1_1_1_0_1/2023-03-28 09-17-53/config.yml"

    controller = TargetTrackingAIController( model_name=model_name, config_name=config_name)
    test_obs = np.linspace(0,18,18)
    #print(test_obs.shape)
    #print(controller.stack_observations(observation_stack=controller.stacked_observations, new_observation=test_obs))
    #print(controller.stacked_observations)
    #controller.set_up(initial_state=[0, 0, 0, 0, 0, 0], target=[1, 1])
    print(controller.get_controller_action(initial_state=[0, 0, 0, 0, 0, 0], target=[1, 1], sample_time=0.2, t =0))
    print(controller.get_controller_action(initial_state=[0, 0, 0, 0, 0, 0], target=[1, 1], sample_time=0.2, t=1))
    print(controller.get_controller_action(initial_state=[0, 0, 0, 0, 0, 0], target=[1, 1], sample_time=0.2, t=3))
    """
    print(model.policy)
    print(model.observation_space)
    print(model.n_steps)
    print(model.action_noise)
    print(model.num_timesteps)
    episodes = 5


    for ep in range(episodes):
        obs = env.reset()
        done = False
        trajectory = np.array([[0],[0], [0]])
        #print(trajectory)
        rewards = 0
        rew_tot = 0
        target = vehicle.target
        rew_trajectory = []

        while not done:
            trajectory = env.get_attr('trajectory')[0] # when using vectorized environments it is reset when done, therfore we must fetch the trajectory before the last run
            #print(obs)
            action, _states = model.predict(obs, deterministic=True )
            #print(f"model prediction returns: {model.predict(obs)}")
            #print(f"action: {action}" )
            #print(f"states: {_states}")
            #print(vehicle.target)
            obs, rewards, done, info = env.step(action)
            rew_trajectory.append(rewards)
            #env.render()
            #print(f"rewards: {rewards}"
            #      f"done: {done}")
            #trajectory = np.append(trajectory, np.reshape(obs[:1, -12:-9][0], (3,1)), axis = 1)
            #print(trajectory)
            #np.set_printoptions(suppress= True, linewidth= 30000)
            rew_tot += rewards
        #print(trajectory.shape)
        print(f"rew_tot: {rew_tot}")
        vehicle = env.get_attr('vehicle')[0]
        plt.plot(list(range(0,len(rew_trajectory))), rew_trajectory, label = 'reward history')
        plt.title(f"run: {ep}")
        plt.show()
        #print(trajectory.shape)
        utils.plot_trajectory(trajectory = trajectory[:, :-1], target= target )
        plt.show()
        """
