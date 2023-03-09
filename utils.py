import time

import matplotlib
#matplotlib.use('tkagg')
from matplotlib import pyplot as plt

from pynput import keyboard
import numpy as np
import casadi as ca
from python_vehicle_simulator.lib import gnc  # some functions from Fossen
import  math

from config import config


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

    """
    if high != low:
        return 2 * (value - low) / (high - low) - 1
    else:
        return 0

def denormalize(low, high, norm_value):
    if norm_value < 0:
        return -low * norm_value
    elif norm_value > 0:
        return high*norm_value
    # (norm_value / 2 + 0.5) * (high - low) + low
    return 0


def get_random_target_point(max_radius):
    theta = np.random.ranf() * 2 * np.pi # radom angle in radians
    r = max_radius * np.sqrt(np.random.ranf()) # find random length of radius within max limit

    x = int(r * np.cos(theta))
    y = int(r * np.sin(theta))
    return [x, y]

def plot_trajectory(trajectory, target=None):
    #print(np.cos(trajectory[2, :]))
    #print(np.sin(trajectory[2, :]))
    #plt.ion()

    plt.close()
    plt.plot(trajectory[0, 0], trajectory[1, 0], label='start', marker='o')
    plt.plot(trajectory[0, :], trajectory[1, :], label='x,y (NED)')

    if target:
        plt.plot(target[0], target[1], marker='x', label='Target')
        circle = plt.Circle((target[0], target[1]), config['env']['target_confident_region'], color='r', fill=False)
        fig = plt.gcf()
        ax = fig.gca()
        ax.add_patch(circle)

    for i in range(trajectory.shape[1]):
        plt.arrow(x=trajectory[0, i], y=trajectory[1, i], dx=np.cos(trajectory[2, i]), dy=np.sin(trajectory[2, i]),length_includes_head=True,
          head_width=0.08, head_length=0.2)

    plt.legend()
    plt.show(block = False)
    plt.pause(1)
    plt.close()


if __name__ == '__main__':

    # example using the normalize function
    print(f"normalize function -10, 10, value 0: {normalize(-10,10, 0)}")
    low= -60
    high= 180
    value = -60
    print(f"normalize function low:{low}, high: {high}, value {value}: {normalize(low=low, high=high, value= value)}")
    print(f"norm and denorm function low: {low}, high: {high}, value {value}: {denormalize(low=low,high=high,norm_value= normalize(low=low, high=high, value=value))}")
    print(f"denorm function low: {low}, high: {high}, value {-0.1}: {denormalize(low=low, high=high, norm_value=0.01)}")

    # example using the point cloud
    point_cloud = []
    for i in range(1000):
        max_radius = 300
        #print(get_random_target_point(max_radius))
        point_cloud.append(get_random_target_point(max_radius))
    point_cloud = np.array(point_cloud)
    print(point_cloud.shape)
    plt.scatter(point_cloud[:, 0], point_cloud[:,1])
    plt.show()
