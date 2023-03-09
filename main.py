#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py: Main program for the Python Vehicle Simulator, which can be used
    to simulate and test guidance, navigation and control (GNC) systems.
"""
import os
import webbrowser
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import mpc.casadi_otter_model_3DOF
import simulator

from python_vehicle_simulator.vehicles import *
from python_vehicle_simulator.lib import *
from otter import otter
from mpc.casadi_otter_model_3DOF import Casadi3dofOtterModel
from mpc.TargetTrackingMPC import TargetTrackingMPC
import utils

# Simulation parameters:
sampleTime = 0.1  # sample time
N = 100  # number of samples

# 3D plot and animation parameters where browser = {firefox,chrome,safari,etc.}
numDataPoints = 50  # number of 3D data points
FPS = 10  # frames per second (animated GIF)
filename = '3D_animation.gif'  # data file for animated GIF
browser = 'safari'  # browser for visualization of animated GIF

###############################################################################
# Vehicle constructors
###############################################################################
printSimInfo()

"""

"""

# no = input("Please enter a vehicle no.: ")
no = '4'
match no:  # the match statement requires Python >= 3.10

    case '3':
        vehicle = otter.otter('headingAutopilot', 0.0, 0.0, 0, 000.0)
    case '4':
        vehicle = otter.otter('TargetTrackingMPC', 0.0, 0.0, 0, 000.0)

printVehicleinfo(vehicle, sampleTime, N)


###############################################################################
# Main simulation loop
###############################################################################
def main():
    # add the functionality to stop the simulation while running
    break_program = False
    target =[10,10]

    otter_6_dof_model = vehicle
    vehicle.target = target
    otter_3_dof_model = Casadi3dofOtterModel(otter_6_dof_model) #implmentere 3DOF uavhengig av 6DOF
    mpc = TargetTrackingMPC(otter_3_dof_model, N=10, sample_time=None)  # N i here prediction horzon for the MPC
    vehicle.set_target_tracking_mpc(mpc)

    def on_press(key):
        if key == keyboard.Key.end:
            # print('end pressed')
            break_program = True
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        while not break_program:
            print('Running simulation...')
            [simTime, simData] = simulator.simulate(N, sampleTime, vehicle)
            print(("sim done"))
            break_program = True  # stop sim
            listener.stop()
        listener.join()

    # print(simData.shape)
    x = simData[:, 0]
    y = simData[:, 1]
    z = simData[:, 2]
    # print(x)
    # print(y)
    # print(z)
    plt.plot(x, y, label = 'x, y (NED)')
    plt.plot(x[0], y[0], label = "Start", marker = 'o')
    plt.plot(target[0], target[1], label = 'target', marker = 'x')
    plotVehicleStates(simTime, simData, 1)
    plotControls(simTime, simData, vehicle, 2)
    plot3D(simData, numDataPoints, FPS, filename, 3)

    """ Uncomment the line below for 3D animation in the web browswer. 
    Alternatively, open the animated GIF file manually in your preferred browser. """
    # webbrowser.get(browser).open_new_tab('file://' + os.path.abspath(filename))

    plt.show()
    # plt.close()


main()
