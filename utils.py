
from pynput import keyboard
import numpy as np
import casadi as ca
from python_vehicle_simulator.lib import gnc  # some functions from Fossen



def on_press(key):
    global break_program
    print(key)
    if key == keyboard.Key.end:
        break_program = True
        return False



def convert_6DOFto3DOF(sixDofMatrix):
    return np.delete(np.delete(sixDofMatrix, (2, 3, 4), 1),(2, 3, 4), 0)


def crossFlowDrags3DOF(L, B, T, nu_r):
    """
    Based on Fossen 2021
    tau_crossflow = crossFlowDrag(L,B,T,nu_r) computes the cross-flow drag
    integrals for a marine craft using strip theory.

    M d/dt nu_r + C(nu_r)*nu_r + D*nu_r + g(eta) = tau + tau_crossflow
    """


    rho = 1026               # density of water
    n = 20                   # number of strips

    dx = L/20
    Cd_2D = gnc.Hoerner(B,T)    # 2D drag coefficient based on Hoerner's curve

    Yh = 0
    Nh = 0
    xL = -L/2

    #print(f"T:{T}, dx: {dx}, Cd_2D: {Cd_2D}, xL:{xL} ")
    #print(f"initial states:  Yh:{Yh},  Nh: {Nh}, xL: {xL}")


    v_r = nu_r[4]  # relative sway velocity
    r = nu_r[5]  # yaw rate


    for i in range(0, n + 1):
        Ucf = ca.fabs(v_r + xL * r) * (v_r + xL * r)
        Yh = Yh - 0.5 * rho * T * Cd_2D * Ucf * dx  # sway force
        Nh = Nh - 0.5 * rho * T * Cd_2D * xL * Ucf * dx  # yaw moment
        xL += dx

    tau_crossflow = -ca.vertcat(0, Yh, Nh)

    #print(f"out:  Yh:{Yh},  Nh: {Nh}, xL: {xL}")
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