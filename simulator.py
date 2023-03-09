#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main simulation loop called by main.py.

Author:     Thor I. Fossen
"""

from pynput import keyboard
import time
import numpy as np
from python_vehicle_simulator.lib import *
from tqdm import tqdm

# from .gnc import attitudeEuler


###############################################################################
# Function printSimInfo(vehicle)
###############################################################################
def printSimInfo():
    """
    Constructors used to define the vehicle objects as (see main.py for details):
        otter('headingAutopilot',psi_d,V_c,beta_c,tau_X)
    """
    print('---------------------------------------------------------------------------------------')
    print('The Python Vehicle Simulator')
    print('---------------------------------------------------------------------------------------')
    print('3 - Otter unmanned surface vehicle (USV): controlled by two propellers, L = 2.0 m')

    print('---------------------------------------------------------------------------------------')


###############################################################################
# Function printVehicleinfo(vehicle)
###############################################################################
def printVehicleinfo(vehicle, sampleTime, N):
    print('---------------------------------------------------------------------------------------')
    print('%s' % (vehicle.name))
    print('Length: %s m' % (vehicle.L))
    print('%s' % (vehicle.controlDescription))
    print('Sampling frequency: %s Hz' % round(1 / sampleTime))
    print('Simulation time: %s seconds' % round(N * sampleTime))
    print('---------------------------------------------------------------------------------------')


###############################################################################
# Function simulate(N, sampleTime, vehicle)
###############################################################################
def simulate(N, sampleTime, vehicle):
    DOF = 6  # degrees of freedom
    t = 0  # initial simulation time

    # Initial state vectors
    eta = np.array([0, 0, 0, 0, 0, 0], float)  # position/attitude, user editable
    nu = vehicle.nu  # velocity, defined by vehicle class
    u_actual = vehicle.u_actual  # actual inputs, defined by vehicle class

    # Initialization of table used to store the simulation data
    simData = np.empty([0, 2 * DOF + 2 * vehicle.dimU], float)

    target = [10, 10]

    # Simulator for-loop
    for i in tqdm(range(0, N + 1)):

        t = i * sampleTime  # simulation time

        # Vehicle specific control systems

        if (vehicle.controlMode == 'headingAutopilot'):
            u_control = vehicle.headingAutopilot(eta, nu, sampleTime)
        elif (vehicle.controlMode == 'TargetTrackingMPC'):
            initial_states = [eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]]
            #print(initial_states)
            u_control = vehicle.target_tracking_mpc(initial_state=initial_states, # 3DOF equation of motion based MPC, therefore the limited parts
                                                    target=None)  # must be a [x,y], if None target from vehicle is used
            print(f"MPC_controls: {u_control}")

        # Store simulation data in simData
        signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)  # original script
        # signals = np.append(np.append(np.append(eta, nu), [0, 0]), [5, 5])

        simData = np.vstack([simData, signals])

        # Propagate vehicle and attitude dynamics
        [nu, u_actual] = vehicle.dynamics(eta, nu, u_actual, u_control, sampleTime)  # velocity is returned
        #print(f"u_actual: {u_actual}")
        eta = attitudeEuler(eta, nu, sampleTime)  # possition!
        # print(t)

    # Store simulation time vector
    simTime = np.arange(start=0, stop=t + sampleTime, step=sampleTime)[:, None]

    return (simTime, simData)
