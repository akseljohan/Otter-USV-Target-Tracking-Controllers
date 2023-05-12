import shutil
import time

import matplotlib
# matplotlib.use('tkagg')
from matplotlib import pyplot as plt
from matplotlib.markers import MarkerStyle

from pynput import keyboard
import numpy as np
import casadi as ca
from python_vehicle_simulator.lib import gnc  # some functions from Fossen
import math

from config import config

plt.rcParams.update({
    "font.family": "serif",
    'lines.linewidth': 1})


def on_press(key):
    global break_program
    print(key)
    if key == keyboard.Key.end:
        break_program = True
        return False


def convert_6DOFto3DOF(sixDofMatrix):
    return np.delete(np.delete(sixDofMatrix, (2, 3, 4), 1), (2, 3, 4), 0)


def crossFlowDrags3DOF(L, B, T, nu_r):
    """
    Based on Fossen 2021
    tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag
    integrals for a marine craft using strip theory.

    M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow
    """

    rho = 1026  # density of water
    n = 20  # number of strips

    dx = L / 20
    Cd_2D = gnc.Hoerner(B, T)  # 2D drag coefficient based on Hoerner's curve

    Yh = 0
    Nh = 0
    xL = -L / 2

    # print(f"T:{T}, dx: {dx}, Cd_2D: {Cd_2D}, xL:{xL} ")
    # print(f"initial states:  Yh:{Yh},  Nh: {Nh}, xL: {xL}")

    v_r = nu_r[4]  # relative sway velocity
    r = nu_r[5]  # yaw rate

    for i in range(0, n + 1):
        Ucf = ca.fabs(v_r + xL * r) * (v_r + xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx  # sway force
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx  # yaw moment
        xL += dx

    tau_crossflow = -ca.vertcat(0, Yh, Nh)

    # print(f"out:  Yh:{Yh},  Nh: {Nh}, xL: {xL}")
    return tau_crossflow


# ------------------------------------------------------------------------------
def m2c(M, nu):
    """
    C = m2c(M,nu) computes the Coriolis and centripetal matrix C from the
    mass matrix M and generalized velocity vector nu (Fossen 2021, Ch. 3)
    """

    M = 0.5 * (M + M.T)  # systematization of the inertia matrix

    # 3-DOF model (surge, sway and yaw)
    # C = [ 0             0            -M(2,2)*nu(2)-M(2,3)*nu(3)
    #      0             0             M(1,1)*nu(1)
    #      M(2,2)*nu(2)+M(2,3)*nu(3)  -M(1,1)*nu(1)          0  ]
    C = ca.MX.zeros(3, 3)
    C[0, 2] = -M[1, 1] * nu[1] - M[1, 2] * nu[2]
    C[1, 2] = M[0, 0] * nu[0]
    C[2, 0] = -C[0, 2]
    C[2, 1] = -C[1, 2]

    return C


def normalize(low, high, value):
    """
    :param low:
    :param high:
    :param value:
    :return:
    """
    if high != low:
        return 2 * (value - low) / (high - low) - 1
    else:
        return 0


def denormalize(low, high, norm_value):
    """
    :param low: lower force value
    :param high: higher force value
    :param norm_value:
    :return: denormalized value
    """
    if norm_value < 0:
        return -low * norm_value
    elif norm_value > 0:
        return high * norm_value
    # (norm_value / 2 + 0.5) * (high - low) + low
    return 0


def get_random_target_point(max_radius):
    """
    :param max_radius:
    :return: x,y
    """
    theta = np.random.ranf() * 2 * np.pi  # radom angle in radians
    r = max_radius * np.sqrt(np.random.ranf())  # find random length of radius within max limit

    x = int(r * np.cos(theta))
    y = int(r * np.sin(theta))
    return [x, y]


def plot_veloceties_ai(trajectory, target=None, file_path=None):
    """
    A method to be used to print the observations from AI in  MarineVehivleTargetTrackingEnv.py
    :param trajectory: np matrix, shape = (12,n) witht he elements [[self.eta[0],  # x (N)
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
                                # the angle between the target and the heading of the vessle
                                self.action[0],
                                self.action[1],
                                self.t / self.time_out,  # time awareness,
                                self.radius  # aware of the region around the target
                                ]]
    :param target: a trajectory with the target coordinates
    :return:
    """

    fig, axis = plt.subplots(4, figsize=(10, 8))

    axis[0].plot(list(range(0, trajectory.shape[1])), trajectory[12, :], label=r'$v_{x_{ref}}$')
    axis[0].set_ylabel('Vel. towards target')
    axis[0].legend()
    axis[1].plot(list(range(0, trajectory.shape[1])), trajectory[10, :], label=r'$d$')
    axis[1].set_ylabel('Euclidean dist.')
    axis[1].legend()
    axis[2].plot(list(range(0, trajectory.shape[1])), trajectory[3, :], label=r'$u$')
    axis[2].plot(list(range(0, trajectory.shape[1])), trajectory[4, :], label=r'$v$')
    axis[2].plot(list(range(0, trajectory.shape[1])), trajectory[5, :], label=r'$r$')
    axis[2].set_ylabel('Velocities')
    axis[2].legend()
    axis[3].plot(list(range(0, trajectory.shape[1])), trajectory[-4, :], label=r'$u_X$')
    axis[3].plot(list(range(0, trajectory.shape[1])), trajectory[-3, :], label=r'$u_N$')
    axis[3].set_ylabel('Actions')
    axis[3].legend()

    if file_path:
        plt.savefig(f'{file_path}/velocity.pdf', pad_inches=0.1, bbox_inches='tight')
    else:
        plt.savefig('velocity.pdf', pad_inches=0.1, bbox_inches='tight')
    plt.show()
    plt.close()


def plot_veloceties(vel_matrix_3DOF, sample_time, file_path=None):
    """
    A method to be used to print the observations from AI in  MarineVehivleTargetTrackingEnv.py
    :param vel_matrix_3DOF: np matrix, shape = (12,n) witht he elements [[self.eta[0],  # x (N)
                                self.eta[1],  # y (E)
                                self.eta[5],  # psi (angle from north)
                                self.nu[0],  # surge vel.
                                self.nu[1],  # sway vel.
                                self.nu[5],  # yaw vel.
                                self.vehicle.target[0],  # x, target
                                self.vehicle.target[1],  # y, target
                                ]]
    :param action_matrix: np matrix, shape(2,n) [[left],
                                                [right]]
    :return:
    """
    t = np.array(list(range(0,len(vel_matrix_3DOF[0])))) * sample_time
    fig, axis = plt.subplots(figsize=(10, 8))

    axis.plot(t, vel_matrix_3DOF[0, :], label=r'$u$ (m/s)')
    #axis[0].legend()
    axis.plot(t, vel_matrix_3DOF[1, :], label=r'$v$ (m/s)')
    #axis[0].legend()
    axis.plot(t, vel_matrix_3DOF[2, :], label=r'$r$ (rad/s)')
    #axis.axhline(0, linestyle="--")
    axis.legend()
    #axis[0].set_ylabel('Velocities')
    #axis.set_yabel('Velocities')
    #plt.ylabel('Velocities')
    #plt.legend()

    if file_path:
        plt.savefig(f'{file_path}', pad_inches=0.1, bbox_inches='tight')
    else:
        plt.savefig('velocity.pdf', pad_inches=0.1, bbox_inches='tight')
    plt.show()
    plt.close()
    return fig


def plot_trajectory(trajectory, target=None, file_path=None):
    """
    Plots the trajectory of the Otter and the target
    :param trajectory: shape() [x,y,psi,u,v,r]
    :param target:
    :return:
    """
    # printing x,y
    fi = plt.figure(figsize=(10, 8))
    ax = plt.subplot()
    ax.scatter(y=trajectory[0, 0], x=trajectory[1, 0], marker='s', s=100)
    ax.plot(trajectory[1, :], trajectory[0, :], label=r'$\eta_{x,y}$')
    # plt.plot(trajectory[6, :], trajectory[7, :], label='target')
    ax.set_aspect('equal', adjustable='box')
    ax.set_ylabel('North')
    ax.set_xlabel('East')

    if target is not None:
        ax.plot(target[1, 0], target[0, 0], marker='x', )
        ax.plot(target[1, :], target[0, :], label=r'$x_{ref}$', linestyle='--', color='orange')
        # circle = plt.Circle((target[0], target[1]), config['env']['target_confident_region'], color='r', fill=False)
        # fig = plt.gcf()
        # ax = fig.gca()
        # ax.add_patch(circle)

    for i in range(trajectory.shape[1]):
        if i % 1000 == 0:
            plt.arrow(trajectory[1, i], trajectory[0, i], dy=np.cos(trajectory[2, i]) * 2,
                      dx=np.sin(trajectory[2, i]) * 2, length_includes_head=False,
                      head_width=3, head_length=3)
            # m = MarkerStyle("triangle")
            # m._transform.rotate_deg(math.degrees(trajectory[2,i]))
        # plt.scatter(x = trajectory[0,i], y = trajectory[1,i],
        #             marker = (4, 1, math.degrees(trajectory[2, i])+180),
        #             color = 'blue')
    plt.legend()
    if file_path:
        plt.savefig(f'{file_path}', pad_inches=0.1, bbox_inches='tight')
    else:
        plt.savefig('NED.pdf', pad_inches=0.1, bbox_inches='tight')
    plt.show(block=False)
    # plt.pause(1)
    plt.close()


def plot_error(pos_otter, pos_target, path=None, sample_time=None):
    """
    :param pos_otter: n,2- matrix x,y values for the otter pos
    :param pos_target: n,2 - np. matrix containing x,y of the target
    :param path: the save path and filename with extension, if non provided no file is saved
    :param sample_time: sampleing time in the data
    :return: fig
    """
    error = np.linalg.norm((pos_otter - pos_target), axis=-1, ord=2)

    fig = plt.figure(figsize=[10, 5])
    ax = plt.subplot()

    ax.plot(np.array(list(range(0, len(error)))) * sample_time, error, label=r'$\||\eta_{x,y} - x_{ref}\||$')
    ax.set_ylabel('Error (m)')
    ax.set_xlabel('Time (s)')
    ax.axhline(0, linestyle="--")
    plt.legend()

    if path:
        plt.savefig(f'{path}', pad_inches=0.1, bbox_inches='tight')
    plt.show()
    plt.close()
    return fig


def plot_controls(u_control, u_actual, sample_time, file_path=None):
    """

    :param u_control: vector for the controls_signals of the Otter
    :param u_actual:  vector with the actual_actuator response of the Otter
    :param sample_time: The sampeling time to derive the time in seconds
    :param file_path: filpeath and filnename for the plots to be saved
    :return: fig
    """
    #fig = plt.figure()
    fig, ax = plt.subplots(1, 2 ,figsize=(10, 8))
    t = np.array(list(range(0,len(u_control)))) * sample_time
    print(t.shape)
    ax[0].set_title('Left')
    ax[0].plot(t, u_control[:,0], label=f'Signal (rad/s)')
    ax[0].plot(t, u_actual[:,0], label=f'Actual (rad/s)')
    ax[0].legend(loc='lower right', bbox_to_anchor=(1.32, -0.12),
                  fancybox=True, shadow=False, ncol=1, borderaxespad = 0.1)
    ax[0].set_xlabel("Time (s)")


    ax[1].set_title('Right')
    ax[1].plot(t, u_control[:,1])#, label=f'Signal (rad/s)')
    ax[1].plot(t, u_actual[:,1])#, label=f'Actual (rad/s)')
    #ax[1].legend(loc='upper right', #bbox_to_anchor=(0.5, -0.005),
    #              fancybox=True, shadow=False, ncol=1, borderaxespad = 0.1)
    ax[1].set_xlabel("Time (s)")
    if file_path is not None:
        plt.savefig(f'{file_path}', pad_inches=0.1, bbox_inches='tight')

    return fig

def plot_solv_time(solv_time_data, sample_time, file_path=None):
    """
    :param tau: control forces
    :param sample_time: sample time of the collected data
    :param file_path: filepath and
    :return:
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    t = np.array(list(range(0, len(solv_time_data)))) * sample_time
    ax.set_ylabel("Seconds")
    ax.plot(t, solv_time_data)
    ax.set_xlabel("Time (s)")
    if file_path is not None:
        plt.savefig(f'{file_path}', pad_inches=0.1, bbox_inches='tight')

    return fig
def plot_control_forces(tau, sample_time, file_path=None):
    """
    :param tau: control forces
    :param sample_time: sample time of the collected data
    :param file_path: filepath and
    :return:
    """
    tau = np.array(tau)
    print(f"plot control forces: input shape {tau.shape}")
    # fig = plt.figure()
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    t = np.array(list(range(0, len(tau)))) * sample_time
    print(t.shape)
    ax[0].set_title('Surge (X)')
    ax[0].plot(t, tau[:, 0])
    #ax[0].legend(loc='lower right', bbox_to_anchor=(1.32, -0.12),
    #             fancybox=True, shadow=False, ncol=1, borderaxespad=0.1)
    ax[0].set_xlabel("Time (s)")

    ax[1].set_title('Yaw (Nm)')
    ax[1].plot(t, tau[:, 1])  # , label=f'Signal (rad/s)')
    # ax[1].legend(loc='upper right', #bbox_to_anchor=(0.5, -0.005),
    #              fancybox=True, shadow=False, ncol=1, borderaxespad = 0.1)
    ax[1].set_xlabel("Time (s)")
    if file_path is not None:
        plt.savefig(f'{file_path}', pad_inches=0.1, bbox_inches='tight')

    return fig
def copy_config(destination_dir):
    shutil.copy(src='config/config.yml', dst=destination_dir)


def simulate_circular_target_motion(initial_position, radius, velocity, sim_time):
    """
    Simulates the position of a particle moving in a circle.

    Args:
    initial_position (Tuple[float, float]): The initial position of the particle as a tuple of x and y coordinates.
    radius (float): The radius of the circle.
    angular_velocity (float): The angular velocity of the particle.
    time (float): The time for which to simulate the particle position.

    Returns:
    Tuple[float, float]:  new position for target.
    """
    angular_velocity = velocity / radius
    angle = angular_velocity * sim_time

    # Calculate the new x and y coordinates
    x = round(initial_position[0] + math.cos(angle) * radius - radius, 4)
    y = round(initial_position[1] + math.sin(angle) * radius, 4)

    return x, y


def simulate_linear_target_motion(initial_position, v_y, v_x, sim_time):
    """
    :param initial_position:
    :param v_y:
    :param v_x:
    :param sim_time:
    :return: new position of target
    """
    x1 = initial_position[0] + (v_x * sim_time)
    y1 = initial_position[1] + (v_y * sim_time)
    return x1, y1


if __name__ == '__main__':

    # example using the normalize function
    print(f"normalize function -10, 10, value 0: {normalize(-10, 10, 0)}")
    low = -60
    high = 180
    value = -60
    print(f"normalize function low:{low}, high: {high}, value {value}: {normalize(low=low, high=high, value=value)}")
    print(
        f"norm and denorm function low: {low}, high: {high}, value {value}: {denormalize(low=low, high=high, norm_value=normalize(low=low, high=high, value=value))}")
    print(f"denorm function low: {low}, high: {high}, value {-0.1}: {denormalize(low=low, high=high, norm_value=0.01)}")

    # example using the point cloud
    point_cloud = []
    for i in range(1000):
        max_radius = 300
        # print(get_random_target_point(max_radius))
        point_cloud.append(get_random_target_point(max_radius))
    point_cloud = np.array(point_cloud)
    print(point_cloud.shape)
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1])
    plt.show()

    x = 0
    y = 0
    v_y = 400
    v_x = 10
    sim_time = 1  # second
    x_traj = []
    y_traj = []
    for i in range(10):
        x, y = simulate_linear_target_motion([0, 0], v_y=v_y, v_x=v_x, sim_time=sim_time)
        print(f"x:{x}, y: {y}")
        x_traj.append(x)
        y_traj.append(y)
    plt.plot(x_traj, y_traj, label="Linear trajectory")
    plt.legend()
    plt.show()
    x_traj = []
    y_traj = []
    N = 100000
    sample_time = 0.5
    for i in (range(0, N + 1)):
        t = i * sample_time
        print(t)
        [x, y] = simulate_circular_target_motion(initial_position=[10, 10], radius=10, velocity=0.25, sim_time=t)
        print(f"x:{x}, y: {y}")
        x_traj.append(x)
        y_traj.append(y)

    print(x_traj)
    plt.plot(x_traj[0], y_traj[0], label="start", marker="x")
    plt.plot(x_traj, y_traj, label="Glider pos in the plane")
    plt.legend()
    plt.show()
