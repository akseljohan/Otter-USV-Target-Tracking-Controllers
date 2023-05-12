import math

import gym
import numpy as np
from gym import spaces
import random
from matplotlib import pyplot as plt
import utils
from config import config
from otter.otter import otter  # modfyed from the original Fossen Vehivle Simulator
from python_vehicle_simulator.lib import gnc  # fossen vehicle simulator library


class TargetTrackingEnv(gym.Env):

    # static variables are defined upon compiling

    def __init__(self):
        super(TargetTrackingEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects #the actions are continuous and normalized

        self.cycle_counter = 0
        self.target_radius = None
        self.within_confidence_radius = False
        self.time_within_confidence_radius = 0
        self.radius = config['env']['target_confident_region']  # defines the region around the target
        self.tota_reward_list = []
        self.time_out = config['env']['time_out']
        self.action_trajectory = None
        self.total_training_time = 0
        self.n_done_tot = 0
        self.psi_d = None
        self.action_dot = 0

        self.action = None
        self.action_space = spaces.Box(low=np.array([-1, -1], dtype=np.float64),
                                       high=np.array([1, 1], dtype=np.float64),
                                       dtype=np.float64)
        # The observations are:
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(18,),
                                            dtype=np.float64)

        self.u_actual = [0, 0]
        self.trajectory = np.array([])
        self.eta = None  # the actual position of the vehicle in the NED-frame
        self.nu = None  # the actual velocity of the vehicle in the BODY-frame
        self.done = None  # record if the training is done or not
        self.vehicle = None  # the otter in this case (must provide a function to calculate the dynamics in the BODY frame)
        self.sample_time = config['sample_time']
        self.sim_length = config['env']['sim_length']
        self.n_steps = range(int(self.sim_length / self.sample_time))
        print(f"sample_time {self.sample_time}")
        self.metadata = {"render.modes": ["human", "plot"]}
        self.boundaries = config['constraints']['forces']  # boundaries
        self.surge_boundaries = self.boundaries['X']
        self.torque_boundaries = self.boundaries['N']
        self.world_box = config['env']['world_size']
        self.previous_actions = []
        self.reward = 0
        self.t = 0
        self.render_mode = config['env']['render_mode']
        self.debug = config['env']['debug']
        self.initial_target_pos = None
        self.target_vel = None

    def __str__(self):
        return str(self.__dict__)

    def step(self, action):
        """
        Step takes in an action from the Agent between. The value are denormalized by utilizing the ranges defined in the config file.
        :param action:
        :return: Observations, stop signal and rewards
        """
        self.action = action
        self.previous_actions.append(action)

        # denormalize action values and calculate truster commands before simulator
        # in this architecture the controller learns the Normalized forces in surge and moment among yaw
        tau_X = utils.denormalize(self.surge_boundaries['lower'], self.surge_boundaries['upper'], action[0])

        tau_N = utils.denormalize(self.torque_boundaries['lower'], self.torque_boundaries['upper'], action[1])

        u_control = np.array(self.vehicle.controlAllocation(
            tau_X=tau_X,
            tau_N=tau_N), float)  # convert forces to control signals (the Agent could learn this directly)

        # this loop is  taken from Fossen vehicle simulator:
        for t_step in self.n_steps:  # by adjusting the values in the config file you can simulate several timesteps with the same controller action.
            self.nu, self.u_actual = self.vehicle.dynamics(eta=self.eta, nu=self.nu,
                                                           u_actual=self.u_actual,
                                                           u_control=u_control,
                                                           sampleTime=self.sample_time)  # The dynamics is the Fossen-  pythonVehicleSimulator
            self.eta = gnc.attitudeEuler(self.eta, self.nu, self.sample_time)

        self.t += 1  # increase time counter
        self.total_training_time += self.sim_length  # increase the total time used in the env

        self.action_dot = (np.linalg.norm(self.previous_actions[-1]) - np.linalg.norm(self.previous_actions[-2])) / len(
            self.n_steps)

        self.vehicle.target = utils.simulate_circular_target_motion(initial_position=self.initial_target_pos,
                                                                    velocity=self.target_vel,
                                                                    radius=self.target_radius,
                                                                    sim_time=self.total_training_time)

        self.psi_d = self.get_psi_d()  # desired heading towards target

        # set observations
        observation = np.array([self.eta[0],  # x (N)
                                self.eta[1],  # y (E)
                                self.eta[5],  # psi (angle from north)
                                self.nu[0],  # surge vel.
                                self.nu[1],  # sway vel.
                                self.nu[5],  # yaw vel.
                                self.vehicle.target[0],  # x, target
                                self.vehicle.target[1],  # y, target
                                self.get_delta_pos_target()[0],  # distance between vehicle and target in x
                                self.get_delta_pos_target()[1],  # distance between vehicle and target in y
                                self.get_euclidean_distance(),  # euclidean distance vehicle and target
                                self.action_dot,
                                self.get_speed_towards_target(),
                                self.get_smalles_sign_angle(angle=(self.get_psi_d() - self.eta[5])),
                                # the angle between the target and the heading of the vessel
                                self.action[0],
                                self.action[1],
                                self.t / self.time_out,  # time awareness,
                                self.radius  # aware of the region around the target
                                ], dtype=np.float64)
        self.trajectory = np.append(self.trajectory, np.array([observation]).T,
                                    axis=1)  # appends a trajectory of observations

        # get rewards
        self.reward, done, euclidean_distance = self.get_reward()

        if self.render_mode == 'human' \
                and abs(self.total_training_time % 10_000) <= self.time_out \
                and self.trajectory.shape[1] > self.time_out:  # check if the episode is finished
            print("Rendering")
            print(f"trajectory shape: {self.trajectory.shape}")
            self.render()

        if self.debug:
            print(f"observations: {observation} \n"
                  f"agent action: {action}\n"
                  f"tau (denormalized action): {[tau_X, tau_N]}\n"
                  f"control signal: {u_control}"
                  f"u_actual:{self.u_actual}\n"
                  f"time: {self.t}\n"
                  f"psi_d: {self.psi_d}\n"
                  f"Smallest sign angle: {self.get_smalles_sign_angle(self.get_psi_d(), self.eta[5])}")

        info = {"observations": observation,
                "agent_action": action,
                "tau": [tau_X, tau_N],
                "u_control": u_control,
                "u_actual": self.u_actual,
                "time": self.t,
                "psi_d": self.psi_d,
                "Smallest sign angle": self.get_smalles_sign_angle(self.get_psi_d(), self.eta[5])}
        return observation, self.reward, done, info

    def reset(self):

        self.previous_actions = [0, 0]
        self.action_trajectory = np.zeros((12,))
        self.action = None
        self.done = False
        self.reward = 0
        self.t = 0
        self.total_reward_list = []
        # reset all values to initial states
        self.eta = [0, 0, 0, 0, 0, 0]
        self.nu = [0, 0, 0, 0, 0, 0]
        self.u_actual = [0, 0]
        self.time_within_confidence_radius = 0  # time within confidence radius
        self.target_radius = config['env']['moving_target']['radius']
        self.total_training_time = 0

        if self.cycle_counter > config['env'][
            'target_cycles'] or self.cycle_counter < 1:  # if it  is the first tiume or the last time in the cycle
            # print(f"cycle_counter: {self.cycle_counter}: {self.cycle_counter > config['env']['target_cycles'] or self.cycle_counter < 1}")
            # define new values for moving or fixed target
            self.cycle_counter = 0
            if config['env']['random_target']:
                self.vehicle.target = utils.get_random_target_point(
                    config['env']['target_spawn_region'])  # define new random target
            # define new target pos, and velocity
            else:
                self.vehicle.target = config['env']['fixed_target']

            self.initial_target_pos = self.vehicle.target  # set new target start point

            vel_range = config['env']['moving_target']['velocity']
            self.target_vel = random.uniform(vel_range[0], vel_range[1])  # set new target velocity

        self.cycle_counter += 1
        self.psi_d = self.get_psi_d()  # get the desired heading
        # self.trajectory = np.zeros(
        #    (self.observation_space.shape[0], 1))  # initialized with zeros so the get speed has an
        observation = np.array([self.eta[0],  # x (N)
                                self.eta[1],  # y (E)
                                self.eta[5],  # psi (angle from north) heading
                                self.nu[0],  # surge vel.
                                self.nu[1],  # sway vel.
                                self.nu[5],  # yaw vel.
                                self.vehicle.target[0],  # x, target
                                self.vehicle.target[1],  # y, target
                                self.get_delta_pos_target()[0],  # distance between vehicle and target in x
                                self.get_delta_pos_target()[1],  # distance between vehicle and target in y
                                self.get_euclidean_distance(),
                                0,  # action_dot
                                0,  # self.get_speed_towards_target(),
                                self.get_smalles_sign_angle(angle=(self.get_psi_d() - self.eta[5])),
                                0,
                                0,
                                0,  # time
                                self.radius  # radius around target
                                ], dtype=np.float64)
        self.trajectory = np.array(np.array([observation]).T)  # initialized with two zeros so that the

        if self.debug:
            print("Env reset")
            print(f"target: {self.vehicle.target}")
            print(f"reset_trajectory:{self.trajectory}")
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        # print(self.trajectory)
        # print(f"trajectory: {self.trajectory}")
        if mode == "human":
            utils.plot_trajectory(self.trajectory, target=self.vehicle.target)

    def close(self):
        None

    def get_psi_d(self):
        """Returns the angle between of the line from the vessel to the target  in NED"""
        return math.atan2(self.vehicle.target[1] - self.eta[1], self.vehicle.target[0] - self.eta[0])

    def get_smalles_sign_angle(self, x=None, y=None, angle=None):  # page 413. in Fossen 2021

        if angle is not None:
            return (angle + math.pi) % (2 * math.pi) - math.pi
        if x is not None and y is not None:
            a = (x - y) % (2 * np.pi)
            b = (y - x) % (2 * np.pi)
            return -a if a < b else b
        else:
            raise ValueError("you must define either x and y, or angle")

    def get_euclidean_distance(self):
        return np.linalg.norm(np.array([self.eta[0], self.eta[1]]) - self.vehicle.target, )

    def get_speed_towards_target(self):

        if self.trajectory.shape[1] > 1:
            pos_k2 = np.array([self.trajectory[0, -2], self.trajectory[1, -2]])
            pos_k1 = np.array([self.trajectory[0, -1], self.trajectory[1, -1]])
            return (np.linalg.norm(pos_k2 - self.vehicle.target) - np.linalg.norm(
                pos_k1 - self.vehicle.target)) / self.sim_length
        else:
            return 0

    def get_delta_pos_target(self):
        delta_x = abs(self.eta[0] - self.vehicle.target[0])
        delta_y = abs(self.eta[1] - self.vehicle.target[1])
        return delta_x, delta_y

    def get_reward(self):
        r = self.radius  # objective area around the target/otter
        a = 1  # amplitude for the reward function
        b = 0  # displacement of bell curve
        c = config['env']['target_spawn_region'] / 2  # the with of the bell-curve
        c1 = config['env']['c1']  # tuning coefficient for distance reward
        c2 = config['env']['c2']  # tuning coefficient for target reached reward
        c3 = config['env']['c3']  # tuning coefficient for controller penalty
        c4 = config['env']['c4']  # tuning coefficient for time penalty
        c5 = config['env']['c5']  # tuning coefficient for time out penalty
        c6 = config['env']['c6']  # tuning coefficient for distance change rate
        c7 = config['env']['c7']  # tuning coefficient for desired heading error
        # initiating the reward and penalty variables
        distance_reward = 0
        target_reached_reward = 0
        controller_action_penalty = 0
        time_penalty = 0
        total_reward = 0
        time_out_penalty = 0
        eucledian_distance = self.get_euclidean_distance()  # (0.5)*(np.tanh(20 * i+2)-1)
        speed_towards_target = c6 * ((np.tanh(3 * self.get_speed_towards_target())))
        heading_reward = -1 + 2 * np.exp(
            -((self.get_smalles_sign_angle(angle=(self.get_psi_d() - self.eta[5]))) ** 2 / (2 * 1) ** 2))
        # (1 * np.exp(-((self.get_smalles_sign_angle(angle=(self.get_psi_d() - self.eta[5]))) ** 2 / (2 * 0.2) ** 2)))

        # reward for reducing distance to target
        distance_reward = (a * np.exp(-((eucledian_distance - b) ** 2 / (
            c) ** 2)))  # gaussian reward function returning a value up reaching the confident area

        # penalties
        # for time usage
        time_penalty = -1  # -self.t / self.time_out #-(-1 + np.exp((self.t / self.time_out) ** 2))  # exponential function that gives a negative revard as a function of the ratio between time and max time
        # controller action penalties
        controller_action_penalty = -abs(self.action_dot)

        # check if target is reached and award agent respectively
        done = False

        # check for termination states (either agent state)
        # check if aget has reach wanted state(Standing relativly calm within the boundary), within the time at hand

        if eucledian_distance < r:
            self.within_confidence_radius = True

        else:
            self.within_confidence_radius = False
            self.time_within_confidence_radius = 0

        if self.within_confidence_radius:  # and -0.5 < self.nu[0] < 0.5 and self.nu[1] < 0.5:
            target_reached_reward += 1
            self.time_within_confidence_radius += 1
            if self.time_within_confidence_radius * self.sample_time > 60:
                # if the agent has been within the radius of confidence more than 60 seconds the wanted state is reached
                print("Reached target for 60 sec!")
                self.done = True
                done = True

        # check if the time has passed
        if self.t >= self.time_out:
            self.done = True
            done = True
            time_out_penalty = -1
            print('time out penalty')

        """#Another function
        if eucledian_distance < r:
            # print("target reached!")
            target_reached_reward += 1
            target_reached = True
            #self.done = True
            #done = True
        if self.t >= self.time_out and eucledian_distance < r:
            # print("time_out!")
            self.done = True
            done = True"""

        total_reward = c1 * distance_reward \
                       + c2 * target_reached_reward \
                       + c3 * controller_action_penalty \
                       + c4 * time_penalty \
                       + c5 * time_out_penalty \
                       + c6 * speed_towards_target \
                       + c7 * heading_reward
        self.total_reward_list = [c1 * distance_reward,
                                  c2 * target_reached_reward,
                                  c3 * controller_action_penalty,
                                  c4 * time_penalty,
                                  c5 * time_out_penalty,
                                  c6 * speed_towards_target,
                                  c7 * heading_reward]
        reward = total_reward
        if self.debug:
            print(f"reward_debug_details:\n"
                  f"pos: {self.eta[0], self.eta[1]} \n"
                  f"heading: {self.eta[2]}\n"
                  f"target at: {self.vehicle.target}\n"
                  f"Total_reward: {reward} \n"
                  f"(distance_reward : {c1 * distance_reward}, speed_towards_target_reward: {c6 * speed_towards_target}, "
                  f"target_reached_reward : {c2 * target_reached_reward},controller_action_penalty: "
                  f"{c3 * controller_action_penalty}, time_penalty: {c4 * time_penalty}, time_out_penalty:"
                  f"{c5 * time_out_penalty} "
                  f":heading_reward: {c7 * heading_reward})"
                  )

        return reward, done, eucledian_distance


if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    np.set_printoptions(linewidth=2000)
    vehicle = otter(0.0, 0.0, 0, 0, 0)
    vehicle.target = [50, 50]
    env = TargetTrackingEnv()
    env.vehicle = vehicle

    print("Checking env, with Stable Baseline 3 env-checker:")
    print(check_env(env))
    env.reset()
    print()
    print("Env initialisation details: ")
    print(env)
    # print(env.reset())
    # print(env.observation_space)
    # It will check your custom environment and output additional warnings if needed
    print("Testing environment")
    for i in range(0, 1):
        env.reset()
        env.vehicle.target = [10, 0]
        temp_rew = []
        temp_dist = []
        done = False
        i = 0
        temp_rew_trajectory_list = []
        while not done:
            obs, rewards, done, info = env.step([1, 0])
            i = 1
            temp_rew.append(rewards)
            temp_rew_trajectory_list.append(env.total_reward_list)
            temp_dist.append(obs[9])
            # print(f"obs_shape: {obs.shape}")
        # env.render()
        trajectory = env.trajectory
        # target = vehicle.target
        print(f"self.initial_target_pos: {env.initial_target_pos}")
        print(f"target vel: {env.target_vel}")
        print(f"target radius: {env.target_radius}")
        print(trajectory[6, :], trajectory[7, :])
        plt.plot(trajectory[6, :], trajectory[7, :], label="target_trajectory")
        plt.legend()
        plt.show()
        fig1 = utils.plot_trajectory(trajectory, target=None)
        c1 = config['env']['c1']  # tuning coefficient for distance reward
        c2 = config['env']['c2']  # tuning coefficient for target reached reward
        c3 = config['env']['c3']  # tuning coefficient for controller penalty
        c4 = config['env']['c4']  # tuning coefficient for time penalty
        c5 = config['env']['c5']  # tuning coefficient for time out penalty
        c6 = config['env']['c6']  # tuning coefficient for distance change rate
        c7 = config['env']['c7']  # tuning coefficient for desired heading error

        fig = plt.figure(figsize=[10, 8])
        ax = plt.subplot(111)
        cycler = plt.cycler(linestyle=[':', ':', ':', ':', ':', ':', ':', '-'],
                            color=['black', 'blue', 'brown', 'orange', 'red', 'green', 'purple', 'olive'])
        ax.set_prop_cycle(cycler)
        lines = ax.plot(list(range(0, len(temp_rew_trajectory_list))), temp_rew_trajectory_list,
                        label=[f'Euclidean distance reward',
                               f'Target reached reward',
                               f'Controller penalty',
                               f'Intermediate time penalty',
                               f'Time out penalty',
                               f'Speed towards target',
                               f'Heading reward']
                        )
        lines = ax.plot(list(range(0, len(temp_rew))), temp_rew, label='Total rewards')
        # plt.plot(list(range(0, len(temp_rew))), temp_dist, label='euclidean distance')

        box = ax.get_position()

        ax.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                  fancybox=True, shadow=False, ncol=2, borderaxespad=0.1)
        ax.set_title('Reward function')
        plt.savefig('rew_func_example.pdf', pad_inches=0.1, bbox_inches='tight')

        plt.show()

        print(f"vehiclen_min/max: {vehicle.n_min}/{vehicle.n_max}")
