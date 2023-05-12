import casadi_otter_model_3DOF
import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
"""
This is a simulator using the 3DOF model and runge kutta integrator to obtain the position in NED and BODY velocities  
"""

# Forward time integration method  -> Integrator to discretize the system using casadi integrators
# x_dot --> x(k+1)

class OtterSimulator:
    def __init__(self, model, N = None, sample_time = None):
        self.x = model.x
        self.u = model.u
        self.ode = model.ode
        self.trajectory = None
        self.N = N
        self.sample_time = sample_time

    def simulate(self, x_initial, u_controls, N, sample_time):
        if not sample_time:
            sample_time = self.sample_time
        if not N:
            N = self.N

        x = self.x
        u = self.u
        ode = self.ode

        #collecting the state, control and ode definitions from the 3DOF-model

        #Integrator definitions and options
        simTime = N * sample_time
        integrator_options = {'tf': sample_time,
                              'simplify': True,
                              'number_of_finite_elements': 4
                              }
        solver = 'rk'  # ipopt' #runge kutta 4

        # DAE problem structure
        f = ca.Function('f_1', [x, u], [ode], ['x', 'u'], ['xdot'])
        dae = {'x': x, 'p': u, 'ode': f(x, u)}


        # oae = ca.Function('ode', [x, u], ode, ['x','u'], ['eta_dot', 'nu_r_dot'] )
        # defining the next state via runge kutta method, provided by Casadi.
        intg = ca.integrator('intg', 'rk', dae, integrator_options)

        x_next = intg(x0=x, p=u)['xf']
        # print(x_next)
        # print(f"x_next= {ca.evalf(intg(x0=x_0, p=u_0)['xf'])}")


        F = ca.Function('F', [x, u], [x_next])
        # print(ca.evalf(F(x_0,u_0)))

        sim = F.mapaccum(N)

        res = sim(x_initial, u_controls)

        res_arr = np.array(ca.evalf(res)) #resulting position and velocities over simulation horizon
        return res_arr

    def write_sim_results(self, res_arr):

        #print(res_arr.shape)
        #print(f"sim time: {simTime} sec")
        n = int(len(res_arr[0]) * 1)
        x = list(range(0, n))

        fig_1 = plt.plot(res_arr[0, :n], res_arr[1, :n], label='x,y (NED)')
        fig_1 = plt.plot(res_arr[0][0], res_arr[1][0], label='Start', marker='o')
        plt.legend()
        plt.show()

        # fig_2 = plt.plot(x, res_arr[2, :n], label='yaw (r)')
        fig_2 = plt.plot(x, res_arr[3, :n], label='surge (m/s)')
        fig_2 = plt.plot(x, res_arr[4, :n], label='sway (m/s)')
        fig_2 = plt.plot(x, res_arr[5, :n], label='yaw rate (r/s)')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    N = 1000
    sample_time = 0.02
    x_initial = [0, 0, 0, 0, 0, 0]  # [x,y,psi,surge,sway,yaw]
    u_controls = [200, 0, 10]  # [force surge, force sway(always zero), torque yaw]
    import otter.otter as otter
    fossen_6_dof_model = otter.otter(0, 0, 0, 0) # Now the 6DOF model is used to extract the M-matrix, the D matrix and other constants.
     #TODO implement a clean variant of Otter_model_3DOF, where otter 6DOF is not used.
    model = casadi_otter_model_3DOF.Casadi3dofOtterModel(fossen_6_dof_otter_model=fossen_6_dof_model)

    sim = OtterSimulator(model = model, N=N, sample_time=sample_time)
    res_arr = sim.simulate(x_initial=x_initial, u_controls=u_controls, N=N, sample_time= sample_time)

    sim.write_sim_results(res_arr=res_arr)