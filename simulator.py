#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is based on the code of:
Main simulation loop called by main.py.

Author:     Thor I. Fossen
"""
import numpy as np
from python_vehicle_simulator.lib import *
from tqdm import tqdm

import utils


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

    initial_vehicle_target = vehicle.target
    print(f"initial_target_pos: {initial_vehicle_target}")
    target_trajectory = []
    tau_trajectory = []
    time_compute_controller_signals = []
    # Simulator for-loop
    for i in tqdm(range(0, N + 1)):

        t = i * sampleTime  # simulation time
        # manipulate target:
        #print(f"t:{t}")
        target = utils.simulate_circular_target_motion(initial_position=initial_vehicle_target, radius=200,velocity=0.25, sim_time=t)
        #vehicle.target = utils.simulate_linear_target_motion()


        start_time = time.time() #start timing how long it takes to compute controller signals
        if (vehicle.controlMode == 'headingAutopilot'):
            u_control = vehicle.headingAutopilot(eta, nu, sampleTime)

        elif (vehicle.controlMode == 'TargetTrackingMPC'):
            #print(f"i: {i}")
            #print(f"i % t*0.5: {t % 0.5}")
            if (t % 0.5) == 0 or t ==0:
                initial_states = [eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]]
                # print(initial_states)

                u_control, tau = vehicle.target_tracking_mpc(initial_state=initial_states,
                                                        # 3DOF equation of motion based MPC, therefore the limited parts
                                                        target=target)  # must be a [x,y], if None target from vehicle is used

        # this is not working as it should, please test the RL-model from the file: AiControllerTesting.py
        elif (vehicle.controlMode == 'TargetTrackingAI'):
            initial_states = [eta[0], eta[1], eta[5], nu[0], nu[1], nu[5]]
            u_control = vehicle.target_tracking_ai_controller(initial_states = initial_states, target = np.array(target) ,sample_time=sampleTime, t= t )


        elif (vehicle.controlMode == 'manual'):
            u_control = [-90, 90]
            #print(f"manual controls:{u_control}")
        # Store simulation data in simData
        end_time = time.time() #end timing how long it takes to compute controller signals
        time_compute_controller_signals.append(end_time - start_time) #log

        signals = np.append(np.append(np.append(eta, nu), u_control), u_actual)  # original script
        # signals = np.append(np.append(np.append(eta, nu), [0, 0]), [5, 5])

        simData = np.vstack([simData, signals])

        # Propagate vehicle and attitude dynamics
        [nu, u_actual] = vehicle.dynamics(eta, nu, u_actual, u_control, sampleTime)  # velocity is returned
        # print(f"u_actual: {u_actual}")
        eta = attitudeEuler(eta, nu, sampleTime)  # possition!
        # print(t)

        #store intermediat sim_data
        target_trajectory.append(target)
        tau_trajectory.append(tau)
    # Store simulation time vector
    time_compute_controller_signals = np.array(time_compute_controller_signals)

    simTime = np.arange(start=0, stop=t + sampleTime, step=sampleTime)[:, None]
    target_trajectory = np.array(target_trajectory)
    #plt.plot(target_trajectory[:,0],target_trajectory[:,1],  label = 'Target trajectory')
    #plt.legend()
    #plt.show()
    return (simTime, simData, target_trajectory,tau_trajectory, time_compute_controller_signals)
