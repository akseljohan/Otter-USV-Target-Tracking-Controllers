import simulator
from otter import otter
import numpy as np
import matplotlib.pyplot as plt
impo

v = otter(0, 0, 0, 0)

            #[eta],[nu],[controls]
[x_next, u_actual] = v.dynamics(eta = [0,0,0,0,0,0],nu = v.nu ,u_control = [50,50], u_actual=[0,0], sampleTime= 0.02 )

print(x_next)


[simTime,simData] = simulator.simulate(1000, 0.02, v)

print(simData.shape)

plt.plot(simTime,simData[:])
plt.show()

plt.plot(simData[:0],simData[:])
plt.show()