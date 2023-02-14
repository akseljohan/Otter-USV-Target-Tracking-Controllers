import casadi as ca
import matplotlib.pyplot as plt

from mpc import casadi_otter_model_3DOF
import numpy as np


class TargetTrackingMPC:

    def __init__(self, ode, x, u, N, sample_time):
        """
        This is the MPC class for the otter takes the model from casadi_otter_model_3DOF
        ode =
        x = The state derivative in the ode
        u = Control inputs (must present in the ODE)
        """
        self.p = None
        self.latest_solv = None
        self.ode = ode
        self.x = x
        self.u = u
        self.current_target = None  # holds the trajectory of the target, relative to the NED-frame
        self.previous_target = None
        self.solv = None
        self.opt_response = None
        self.opt_controls = None
        self.solv_target = None
        self.target = None
        self.N = N
        self.sample_time = sample_time
        self.opti = self.initiate_MPC()

        # initiate the MPC

    def set_target(self, target):
        self.opti.set_value(self.target, target)


    def initiate_MPC(self):

        # define local variables_
        N = self.N
        sample_time = self.sample_time
        # Define integrator for #next_state = x_next
        # Integrator definitions and options
        integrator_options = {'tf': sample_time,
                              'simplify': True,
                              'number_of_finite_elements': 4
                              }
        solver = 'rk'  # ipopt' #runge kutta 4

        # DAE problem structure
        f = ca.Function('f_1', [self.x, self.u], [ode], ['x', 'u'], ['xdot'])
        dae = {'x': self.x, 'p': self.u, 'ode': f(self.x, self.u)}

        # oae = ca.Function('ode', [x, u], ode, ['x','u'], ['eta_dot', 'nu_r_dot'] )
        # defining the next state via runge kutta method, provided by Casadi.
        intg = ca.integrator('intg', 'rk', dae, integrator_options)

        x_next = intg(x0=self.x, p=self.u)['xf']
        # print(x_next)
        # print(f"x_next= {ca.evalf(intg(x0=x_0, p=u_0)['xf'])}")

        F = ca.Function('F', [self.x, self.u], [x_next])  # function for the next state

        # define optimization environment:
        opti = ca.Opti()
        self.opti = opti

        # Define optimization paramaters:
        """Q = ca.MX(ca.vertcat(ca.horzcat(1, 0),
                             ca.horzcat(0, 1)))
        R = ca.MX(ca.vertcat(ca.horzcat(1, 1),
                             ca.horzcat(0, 1)))"""
        # cf= # cost function

        x = opti.variable(6, N + 1)  # we only return one state-matrix that contains eta, and nu, after discretization
        u = opti.variable(3, N + 1)  # control variable
        p = opti.parameter(6)  # used to fix the first state
        target = opti.parameter(2)
        Q = opti.parameter(2, 2)
        R = opti.parameter(3, 3)
        cf = 0
        # define cost function
        for k in range(N):
            # opti.subject_to(x[:,k+1] == F(x[:, k], u[:, k]))
            cf = cf + (((x[:2, k] - target).T @ Q @ (x[:2, k] - target)) + \
                       (u[:, k].T @ R @ u[:, k]))

        cost_function = ca.Function('cost_function', [x, u, target], [cf])

        # create an optimization environment with a Multiple shooting discretization,
        # that means that the whole state and control trajectory is the decition variables

        opti.minimize(cost_function(x, u, target))  # objective function this is primarily the distance between the state and the reference (target)
        # opti.subject_to(x == F(x, u))

        for k in range(N):
            opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k]))

        opti.subject_to(u[0, :] < 200)  # surge max 200N
        opti.subject_to(u[0, :] > -50)  # surge minimum 200N
        opti.subject_to(u[1, :] == 0)  # yaw == 0 always
        # opti.subject_to(u[1:] > -0.1)  # yaw == 0 always
        opti.subject_to(u[2, :] < 50)  # max 50 Nm torque among yaw
        opti.subject_to(u[2, :] > -50)  # min 50Nm torque among yaw
        opti.subject_to(x[:, 0] == p)  # initial states must always be the initial states

        opti.solver('ipopt')
        opti.set_value(Q, np.diag([1, 1]))
        opti.set_value(R, np.diag([10, 0, 10]))

        self.p = p
        self.x = x
        self.u = u
        self.target = target

        return opti

    def solve_optimal_control(self, initial_state, target):
        if target:
            self.opti.set_value(self.target, target)

        self.opti.set_value(self.p, initial_state)
        solv = self.opti.solve()
        self.latest_solv = solv

        return solv.value(self.u)

    def write_latest_results(self):
        opt_controls = self.latest_solv.value(self.u) #collects the numerical values from the solve-object
        opt_response = self.latest_solv.value(self.x)
        solv_target = self.latest_solv.value(self.target)
        print(f"sim time: {self.N * self.sample_time} sec")
        n = int(len(opt_controls[0]) * 1)
        x = list(range(0, n + 1))
        fig_1 = plt.plot(opt_response[0, :n], opt_response[1, :n], label='x,y (NED)')
        fig_1 = plt.plot(opt_response[0][0], opt_response[1][0], label='Start', marker='o')
        fig_1 = plt.plot(solv_target[0], solv_target[1], label='Target', marker='x')
        plt.legend()
        plt.show()

        fig_2, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
        ax1.stairs(values=opt_controls[0], edges=x, label='Surge (N)')
        ax1.legend()
        ax2.stairs(values=opt_controls[1], edges=x, label='Sway (N)')
        ax2.legend()
        ax3.stairs(values=opt_controls[2], edges=x, label='Yaw (Nm)')
        ax3.legend()
        fig_2.show()


if __name__ == '__main__':
    """
    Example of usage:
    
    """
    # collecting the state, control and ode definitions from the 3DOF-model
    x = casadi_otter_model_3DOF.x
    u = casadi_otter_model_3DOF.u
    ode = casadi_otter_model_3DOF.ode

    MPC = TargetTrackingMPC(ode=ode, x=x, u=u, N=10, sample_time=0.02)
    #MPC.set_target(n)
    control_force = MPC.solve_optimal_control(initial_state=[0, 0, 0, 0, 0, 0], target= [1, 2] )
    print(control_force[:,0])
    #MPC.write_latest_results()
