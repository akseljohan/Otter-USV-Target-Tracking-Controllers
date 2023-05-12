#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py: Main program for the Python Vehicle Simulator, which can be used
    to simulate and test guidance, navigation and control (GNC) systems.
"""
import os
# import webbrowser
# import matplotlib
# matplotlib.use("TkAgg")
# import matplotlib.pyplot as plt
import pandas as pd
# import mpc.casadi_otter_model_3DOF
import simulator

# from python_vehicle_simulator.vehicles import *
from python_vehicle_simulator.lib import *

from AI_controller.TargetTrackingAIController import TargetTrackingAIController
from otter import otter
from mpc.casadi_otter_model_3DOF import Casadi3dofOtterModel
from mpc.TargetTrackingMPC import TargetTrackingMPC
import utils
from config import config
from datetime import datetime
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    'lines.linewidth': 1})

# Simulation parameters:
sampleTime = config['sample_time']  # sample time
N = 8000  # number of samples

# 3D plot and animation parameters where browser = {firefox,chrome,safari,etc.}
# numDataPoints = 50  # number of 3D data points
# FPS = 10  # frames per second (animated GIF)
# filename = '3D_animation.gif'  # data file for animated GIF
# browser = 'safari'  # browser for visualization of animated GIF

###############################################################################
# Vehicle constructors
###############################################################################
printSimInfo()
# select 1 for MPC test, or '2' for AI test
no = '1'
match no:  # the match statement requires Python >= 3.10

    case '1':
        vehicle = otter.otter('TargetTrackingMPC', 0.0, 0.0, 0, 000.0)
        otter_6_dof_model = vehicle
        otter_3_dof_model = Casadi3dofOtterModel(otter_6_dof_model)  # implmentere 3DOF uavhengig av 6DOF
        mpc = TargetTrackingMPC(otter_3_dof_model, N=config['MPC']['prediction_horizon'],
                                sample_time=None)  # N is the prediction horizon for the MPC
        vehicle.set_target_tracking_mpc(mpc)
    case '2':
        vehicle = otter.otter('TargetTrackingAI', 0.0, 0.0, 0, 000.0)

        # f"AI_controller/logs/PPO_end_when_reaching_target_true/500_150_5_True_0_1_1_1_0_1/2023-03-28 09-17-53/best_model.zip" #this is not done training
        model_dir = f"AI_controller/logs/PPO_moving_target_normalized/1000_30_1_True_0_2_1_1_0_1/2023-04-09 17-17-17/"
        # f"AI_controller/logs/PPO_moving_target_transferred_lr/1000_30_1_True_0_2_1_1_0_1/2023-04-08 18-32-40/"
        # f"AI_controller/logs/PPO_moving_target_transferred_lr/1000_30_1_True_0_2_1_1_0_1/2023-04-08 17-18-52/"
        # f"AI_controller/logs/PPO_end_when_reaching_target_true/500_150_5_True_0_1_1_1_0_1/2023-03-28 09-17-53/"
        # f"AI_controller/logs/PPO_end_when_reaching_target_true/500_150_5_True_0_1_1_1_0_1/2023-03-27 08-03-59/"#this is slow
        model_name = model_dir + f"best_model.zip"
        config_name = model_dir + f"config.yml"
        # f"AI_controller/logs/PPO_end_when_reaching_target_true/500_150_5_True_0_1_1_1_0_1/2023-03-27 08-03-59/config.yml"
        # f"AI_controller/logs/PPO_end_when_reaching_target_true/500_150_5_True_0_1_1_1_0_1/2023-03-28 09-17-53/config.yml" #this is not converged
        ai_controller = TargetTrackingAIController(model_name=model_name, config_name=config_name)
        vehicle.set_target_tracking_ai_controller(ai_controller=ai_controller)
    case '5':
        vehicle = otter.otter('manual', 0.0, 0.0, 0, 000.0)
        print('manual_controls')
printVehicleinfo(vehicle, sampleTime, N)


###############################################################################
# Main simulation loop
###############################################################################
def main():
    # add the functionality to stop the simulation while running
    break_program = False
    target = [20,20]
    vehicle.target = target

    result_dir = os.path.join('results', vehicle.controlMode,
                              datetime.now().strftime('%Y-%m-%d %H-%M-%S'))  # create a log dir for the results
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    utils.copy_config(result_dir)  # copy the config file for documentation

    def on_press(key):
        if key == keyboard.Key.end:
            # print('end pressed')
            break_program = True
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        while not break_program:
            print('Running simulation...')
            simTime, simData, target_trajectory, tau_trajectory, time_compute_controller_signals = simulator.simulate(N, sampleTime, vehicle)
            print(("sim done"))
            break_program = True  # stop sim
            listener.stop()
        listener.join()

    # save the data in teh result (for future treatment of plots and research)
    np.savetxt(fname=result_dir + '/sim_time.csv', X=simTime, delimiter=",", fmt="%f")
    np.savetxt(fname=result_dir + '/simData.csv', X=simData, delimiter=",", fmt="%f")
    np.savetxt(fname=result_dir + '/target_trajectory.csv', X=target_trajectory, delimiter=",", fmt="%f")
    np.savetxt(fname=result_dir + '/time_controller_computation.csv', X=time_compute_controller_signals, delimiter=",", fmt="%f")
    np.savetxt(fname=result_dir + '/tau_trajectory.csv', X=tau_trajectory, delimiter=",", fmt="%f")

    # plot target and vehicle trajectory
    pos_3dof = simData[:, [0, 1, 5, 6, 7, 11]]
    utils.plot_trajectory(trajectory=pos_3dof.T, target=target_trajectory.T,
                          file_path=f'{result_dir}/NED.pdf')

    # plot error plot
    utils.plot_error(simData[:, [0, 1]], pos_target=target_trajectory, path=f'{result_dir}/error.pdf',
                     sample_time=sampleTime)

    # plot the controller signals
    utils.plot_controls(u_control=simData[:, [12, 13]],
                        u_actual=simData[:, [14, 15]],
                        sample_time=sampleTime,
                        file_path=f'{result_dir}/control_rpms.pdf')

    utils.plot_veloceties(vel_matrix_3DOF=simData[:, [6,7,11]].T,
                          action_matrix=simData[:, [12, 13]].T,
                          sample_time=sampleTime,
                          file_path=f'{result_dir}/velocities.pdf')

    utils.plot_control_forces(tau_trajectory[[0,2],:], sample_time=sampleTime, file_path=f'{result_dir}/control_forces.pdf')
    plt.show()

    utils.plot_solv_time(solv_time_data=time_compute_controller_signals, sample_time=sampleTime, file_path=f'{result_dir}/control_compute_time.pdf')
    plt.show()
main()
