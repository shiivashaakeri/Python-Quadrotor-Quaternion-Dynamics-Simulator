import numpy as np
from scipy.integrate import solve_ivp

from utils.quaternion_utils import QuaternionUtils


class RocketDynamics:
    def __init__(self, params):
        """
        Initialize the rocket dynamics with system parameters.

        Parameters:
        params (dict): Dictionary containing system parameters such as mass, gravity, inertia tensor, etc.
        """
        self.params = params

    def nonlinear_dynamics(self, t, state, control_current, control_slope, t_start, prop):
        """

        Parameters are the same as before.
        """
        # Ensure control_slope is never None
        if control_slope is None:
            control_slope = np.zeros_like(control_current)
        # Compute control input
        if self.params["dis_type"] == "FOH" and self.params["dis_exact"] and not prop:
            control = control_current + control_slope * (t - t_start)
        else:
            control = control_current

        # Extract state variables
        r_i = state[:3]  # Position  # noqa: F841
        v_i = state[3:6]  # Velocity
        q_bi = state[6:10]  # Quaternion
        w_b = state[10:]  # Angular rate

        # Extract control inputs
        u_mB = control[:3]  # Applied force
        tau_i = control[3:]  # Applied torque

        # Normalize quaternion
        q_bi /= np.linalg.norm(q_bi)

        # Compute dynamics
        p_i_dot = v_i
        v_i_dot = (1 / self.params["m"]) * QuaternionUtils.qdcm(q_bi) @ u_mB + np.array([0, 0, self.params["g"]])
        q_bi_dot = 0.5 * QuaternionUtils.skew_symmetric_matrix_quat(w_b) @ q_bi
        w_b_dot = np.diag(1 / self.params["J_b"]) @ (
            tau_i - QuaternionUtils.skew_symmetric_matrix(w_b) @ np.diag(self.params["J_b"]) @ w_b
        )

        return np.hstack([p_i_dot, v_i_dot, q_bi_dot, w_b_dot])

    def augmented_dynamics(self, t, state, control_current, control_slope, t_start, prop, obstacles):
        """
        Compute the augmented dynamics of the rocket with CTCS augmentation for obstacles.

        Parameters are the same as before, with the addition of `obstacles`.
        """
        # Compute control input
        if self.params["dis_type"] == "FOH" and self.params["dis_exact"] and not prop:
            control = control_current + control_slope * (t - t_start)
        else:
            control = control_current

        # Extract state variables
        r_i = state[:3]  # Position
        v_i = state[3:6]  # Velocity
        q_bi = state[6:10]  # Quaternion
        w_b = state[10:13]  # Angular rate

        # Compute CTCS augmentation
        g_dot = []
        for obs in obstacles:
            if self.params["ctcs"]:
                g_dot.append(np.maximum(0, obs.g_bar_ctcs(r_i)) ** 2)
            else:
                g_dot.append(0)
        g_dot = np.array(g_dot)

        # Extract control inputs
        u_mB = control[:3]  # Applied force
        tau_i = control[3:]  # Applied torque

        # Normalize quaternion
        q_bi /= np.linalg.norm(q_bi)

        # Compute dynamics
        p_i_dot = v_i
        v_i_dot = (1 / self.params["m"]) * QuaternionUtils.qdcm(q_bi) @ u_mB + np.array([0, 0, self.params["g"]])
        q_bi_dot = 0.5 * QuaternionUtils.skew_symmetric_matrix_quat(w_b) @ q_bi
        w_b_dot = np.diag(1 / self.params["J_b"]) @ (
            tau_i - QuaternionUtils.skew_symmetric_matrix(w_b) @ np.diag(self.params["J_b"]) @ w_b
        )

        state_dot = np.hstack([p_i_dot, v_i_dot, q_bi_dot, w_b_dot])
        augmented_state_dot = np.hstack([state_dot, g_dot])
        return augmented_state_dot

    def simulate_nonlinear(self, initial_state, controls, dt):
        """
        Simulate the nonlinear dynamics of the rocket.

        Parameters:
        initial_state (np.ndarray): Initial state vector of the system.
        controls (np.ndarray): Control input matrix (n_controls x time steps).
        dt (float): Simulation time step.

        Returns:
        np.ndarray: Array of states over time.
        """
        states = [initial_state]
        controls_next = None
        controls_slope = None

        # Handle the case where controls are a single control vector
        if controls.shape == (self.params["n_controls"],):
            sol = solve_ivp(self.nonlinear_dynamics, (0, dt), initial_state, args=(controls, controls_next, dt, False))
            return sol.y[:, -1]

        # Multi-step simulation
        t_eval_opt = np.arange(0, self.params["total_time"] - self.params["dt_ss"], self.params["dt_ss"])

        for i, t in enumerate(t_eval_opt):
            controls_current = controls[:, i]

            # Compute control slope for FOH (First Order Hold) if applicable
            if self.params["dis_type"] == "FOH" and self.params["dis_exact"]:
                controls_next = controls[:, i + 1]
                controls_slope = (controls_next - controls_current) / self.params["dt_ss"]

            # Generate evaluation time points for the current time step
            t_eval = np.arange(t, t + self.params["dt_ss"] + 1e-10, self.params["dt_sim"])

            # Concatenate evaluation points for the entire simulation
            if i == 0:  # noqa: SIM108
                t_eval_full = t_eval
            else:
                t_eval_full = np.hstack([t_eval_full, t_eval])

            # Solve the dynamics for the current interval
            sol = solve_ivp(
                self.nonlinear_dynamics,
                (t, t + self.params["dt_ss"] + 1e-10),
                states[-1],
                args=(controls_current, controls_slope, t, False),
                t_eval=t_eval,
                rtol=1e-3,
                atol=1e-3,
                method="DOP853",  # Use a high-accuracy solver
            )

            # Add all states from the solution
            for k in range(1, sol.y.shape[1]):
                states.append(sol.y[:, k])

        return np.array(states)

    def simulate_nonlinear_interval(self, initial_state, controls, dt, interval):
        """
        Simulate the nonlinear dynamics of the rocket over a given time interval.

        Parameters:
        initial_state (np.ndarray): Initial state vector of the system.
        controls (np.ndarray): Control input matrix (n_controls x time steps).
        dt (float): Simulation time step.
        interval (tuple): (t_start, T) specifying the start and end of the interval.

        Returns:
        np.ndarray: Array of states within the interval.
        """
        states = [initial_state]
        controls_next = None
        controls_slope = None

        # Extract start and end times from interval
        t_start, T = interval

        # Generate time steps similar to simulate_nonlinear
        t_eval_opt = np.arange(t_start, T - self.params["dt_ss"] + 1e-10, self.params["dt_ss"])

        for i, t in enumerate(t_eval_opt):
            controls_current = controls[:, i]

            # Compute control slope for FOH (First Order Hold) if applicable
            if self.params["dis_type"] == "FOH" and self.params["dis_exact"]:
                controls_next = controls[:, i + 1]
                controls_slope = (controls_next - controls_current) / self.params["dt_ss"]

            # Generate evaluation time points for the current time step
            t_eval = np.arange(t, t + self.params["dt_ss"] + 1e-10, dt)

            # Concatenate evaluation points
            if i == 0:  # noqa: SIM108
                t_eval_full = t_eval
            else:
                t_eval_full = np.hstack([t_eval_full, t_eval])

            # Solve the dynamics for the current interval
            sol = solve_ivp(
                self.nonlinear_dynamics,
                (t, t + self.params["dt_ss"] + 1e-10),
                states[-1],
                args=(controls_current, controls_slope, t, False),
                t_eval=t_eval,
                rtol=1e-3,
                atol=1e-3,
                method="DOP853",
            )

            # Add all states from the solution
            for k in range(1, sol.y.shape[1]):
                states.append(sol.y[:, k])

        return np.array(states)
