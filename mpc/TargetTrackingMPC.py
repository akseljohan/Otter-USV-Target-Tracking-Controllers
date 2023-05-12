import casadi as ca
import matplotlib.pyplot as plt

from config import config
from mpc import casadi_otter_model_3DOF
import numpy as np


class TargetTrackingMPC:

    def __init__(self, model, N, sample_time=None, solver=None):
        """
        This is the MPC class for the otter takes the model from casadi_otter_model_3DOF
        The model must provide:
            ode = casadi_otter_model_3DOF.ode
            x = Casadi symobl defined in the state derivative in the ode
            u = Control inputs (must present in the ODE)
        inspiration: https://web.casadi.org/blog/ocp/
        """
        self.model = model
        self.p = None
        self.latest_solv = None
        self.ode = model.ode
        self.x = model.x
        self.u = model.u
        self.current_target = None  # holds the trajectory of the target, relative to the NED-frame
        self.previous_target = None
        self.solv = None
        self.opt_response = None
        self.opt_controls = None
        self.solv_target = None
        self.target = None
        self.N = N

        if not sample_time:
            self.sample_time = config['MPC']['sample_time']
        else:
            self.sample_time = sample_time

        self.solver = solver
        self.solver_options = None

        self.opti = self.__initiate_mpc()

        self.set_solver()
        print(f"solver and options set to: {self.get_solver_options()}")
        # initiate the MPC

    def set_target(self, target):
        self.opti.set_value(self.target, target)

    def set_solver(self, solver: str):
        self.solver = solver

    def __initiate_mpc(self):

        # define local variables_
        ode = self.ode
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
        I = opti.parameter(3,3)
        cf = 0
        # define cost function
        # for k in range(N): #multiple shooting method
        #    # opti.subject_to(x[:,k+1] == F(x[:, k], u[:, k]))
        #    cf = (((x[:2, k] - target).T @ Q @ (x[:2, k] - target)) + \
        #               (u[:, k].T @ R @ u[:, k]))
        # cf = (ca.sumsqr((x[:2, :] - target).T @ Q @ (x[:2, :] - target)) +  (ca.sumsqr(u.T @ R @ u)))


        for k in range(N):
            opti.subject_to(x[:, k + 1] == F(x[:, k], u[:, k])) #to repect the dynamics of the system
            cf = cf +((x[:2, k] - target).T @ Q @ (x[:2, k] - target) + ((u[:, k]-u[:, k-1]).T @ I @ (u[:, k]-u[:, k+1]))+(u[:, k]).T @ R @ (u[:, k]))
            #cf = ((x[:2, k] - target).T @ Q @ (x[:2, k] - target) + ((u[:, k] - u[:, k - 1]).T @ I @ (u[:, k] - u[:, k - 1])) + ca.eig((u[:, k]).T @ R))
            #cf = ((ca.norm_2((x[:2, k] - target).T  @ Q)) + (ca.norm_2((u[:, k]).T  @ R)))
            # cf = (((x[:2, k] - target).T @ Q @ (x[:2, k] - target)) + ((u[:, k] -u[:, k+1]).T @ R @ (u[:, k] -u[:, k+1]))) #delta u

        cost_function = ca.Function('cost_function', [x, u, target,R,Q], [cf])

        # create an optimization environment with a Multiple shooting discretization,
        # that means that the whole state and control trajectory is the decition variables

        opti.minimize(cost_function(x, u, target, R, Q))  # objective function this is primarily the distance between the state and the reference (target)
        # opti.subject_to(x == F(x, u))

        opti.subject_to(opti.bounded(-150, u[0, :], 150))  # surge lower and upper bound
        opti.subject_to(u[1, :] == 0)  # yaw controls == 0 always
        opti.subject_to(opti.bounded(-80, u[2, :], 80))  # max 50 Nm torque among yaw
        opti.subject_to(x[:, 0] == p)  # initial states must always be the initial states
        opti.set_value(Q, np.diag(config['MPC']['Q']))
        opti.set_value(R, np.diag(config['MPC']['R']))
        opti.set_value(I, np.diag([1,1,1]))
        # solver_options = {"ipopt": {"max_iter": 10, "print_level": 5}}



        self.p = p
        self.x = x
        self.u = u
        self.target = target

        return opti

    def get_solver_options(self):
        options = config['MPC']['nlp_options']
        options[config['MPC']['solver']] = config['MPC']['options']['ipopt']
        return options

    def set_solver(self):
        self.solver = config['MPC']['solver']
        options = self.get_solver_options()
        if options:
            self.opti.solver(self.solver, self.get_solver_options())
        else:
            self.opti.solver(self.solver)

    def solve_optimal_control(self, initial_state, target):

        if target:
            self.opti.set_value(self.target, target)
        if config['MPC']['debug']:
            self.opti.debug.value
        #print(f"target: {target}")
        self.opti.set_value(self.p, initial_state)
        # self.opti.set_initial(self.x, self.x)
        if self.latest_solv is not None:
            #self.opti.set_initial(self.x, self.latest_solv.value(self.x))
            self.opti.set_initial(self.u, self.latest_solv.value(self.u))
        else:
            self.opti.set_initial(self.u, 0)
        solv = self.opti.solve()
        #print(solv.stats())
        #print(solv)
        self.latest_solv = solv


        return solv.value(self.u)[:, 0]  # only returns the first in the trajectory

    def write_latest_results(self):
        opt_controls = self.latest_solv.value(self.u)  # collects the numerical values from the solve-object
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

    def __str__(self) -> str:
        return f"solver: {self.solver}" \
               f"Prediction horizon: {self.N}" \
               f"sampleing time : {self.sample_time}"


if __name__ == '__main__':
    """
    Example of usage:
    
    """
    # collecting the state, control and ode definitions from the 3DOF-model
    import otter.otter as otter

    fossen_6_dof_model = otter.otter(0, 0, 0,
                                     0)  # Now the 6DOF model is used to extract the M-matrix, the D matrix and other constants.

    # TODO implement a clean variant of Otter_model_3DOF, where otter 6DOF is not used.
    model = casadi_otter_model_3DOF.Casadi3dofOtterModel(fossen_6_dof_otter_model=fossen_6_dof_model)

    MPC = TargetTrackingMPC(model=model, N=100, sample_time=0.2)
    # MPC.set_target(n)
    control_force = MPC.solve_optimal_control(initial_state=[0, 0, 0, 0, 0, 0], target=[10, 10])
    # print(control_force[:, 0])
    MPC.write_latest_results()
