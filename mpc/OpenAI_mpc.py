import numpy as np
from scipy.optimize import minimize


# Fossen equation of motion in 3DOF
def eq_of_motion(state, control, params):
    x, y, psi, u, v, r = state
    n_x, n_y, n_psi = control
    m, J_z, X_u, Y_v, N_r = params
    dx = np.array([u * np.cos(psi) - v * np.sin(psi),
                   u * np.sin(psi) + v * np.cos(psi),
                   r,
                   (n_x + Y_v * r - X_u * v) / m,
                   (n_y + X_u * r + Y_v * u) / m,
                   (N_r + J_z * r) / J_z])
    return dx


# Model Predictive Controller
def MPC(target_position, initial_state, params, horizon=10, dt=0.1):
    x_tgt, y_tgt, psi_tgt = target_position
    x, y, psi, u, v, r = initial_state
    m, J_z, X_u, Y_v, N_r = params

    def objective(controls):
        state = np.array(initial_state)
        for i in range(horizon):
            state += eq_of_motion(state, controls[i * 3:(i + 1) * 3], params) * dt
        x_err = x_tgt - state[0]
        y_err = y_tgt - state[1]
        psi_err = psi_tgt - state[2]
        return x_err ** 2 + y_err ** 2 + psi_err ** 2

    control_init = np.zeros((horizon * 3,))
    control_bounds = [(None, None)] * (horizon * 3)
    result = minimize(objective, control_init, bounds=control_bounds)
    return result.x[:3]


# Example usage
target_position = [10, 20, np.pi / 4]
initial_state = [5, 15, np.pi / 8, 1, 2, 0.5]
params = [100, 50, 0.1, 0.2, 0.3]
control = MPC(target_position, initial_state, params)
print(control)