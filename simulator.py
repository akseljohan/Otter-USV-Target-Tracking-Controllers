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
    eta = np.array([0, 0, 0.2, 0, 0, 0], float)  # position/attitude, user editable
    nu = vehicle.nu  # velocity, defined by vehicle class
    u_actual = vehicle.u_actual  # actual inputs, defined by vehicle class

    # Initialization of table used to store the simulation data
    simData = np.empty([0, 2 * DOF + 2 * vehicle.dimU], float)

    # Simulator for-loop
    for i in range(0, N + 1):

        t = i * sampleTime  # simulation time

        # Vehicle specific control systems
        if (vehicle.controlMode == 'depthAutopilot'):
            u_control = vehicle.depthAutopilot(eta, nu, sampleTime)
        elif (vehicle.controlMode == 'headingAutopilot'):
            u_control = vehicle.headingAutopilot(eta, nu, sampleTime)
        elif (vehicle.controlMode == 'depthHeadingAutopilot'):
            u_control = vehicle.depthHeadingAutopilot(eta, nu, sampleTime)
        elif (vehicle.controlMode == 'DPcontrol'):
            u_control = vehicle.DPcontrol(eta, nu, sampleTime)
        elif (vehicle.controlMode == 'stepInput'):
            u_control = vehicle.stepInput(t)

            # Store simulation data in simData
        signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)

        simData = np.vstack([simData, signals])

        # Propagate vehicle and attitude dynamics
        [nu, u_actual] = vehicle.dynamics(eta, nu, u_actual, u_control, sampleTime)
        eta = attitudeEuler(eta, nu, sampleTime)  # possition!
        print(t)

    # Store simulation time vector
    simTime = np.arange(start=0, stop=t + sampleTime, step=sampleTime)[:, None]

    return (simTime, simData)
