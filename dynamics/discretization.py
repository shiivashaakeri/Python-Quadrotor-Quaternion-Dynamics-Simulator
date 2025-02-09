import numpy as np
import scipy.linalg
from scipy.integrate import solve_ivp

from params import rocket_params
from utils.quaternion_utils import QuaternionUtils


class Discretization:
    def __init__(self, params):
        """
        Initialize the Discretization class with system parameters.

        Parameters:
        params (dict): Parameters of the system, including the number of states, controls, and other discretization-related parameters.
        """
        self.params = params

    def zoh_discretized_dynamics(self, dt: float, state: np.ndarray, control: np.ndarray, A, B):
        """
        Zero-order hold (ZOH) discretized dynamics.

        Parameters:
        dt (float): Discretization time step.
        state (np.ndarray): Current state vector.
        control (np.ndarray): Current control vector.
        A, B: Jacobian functions with respect to state and control.

        Returns:
        A_d, B_d: Discretized state and control matrices.
        """
        p = np.expand_dims(state[:3], 1)
        v = np.expand_dims(state[3:6], 1)
        q = np.expand_dims(state[6:10], 1)
        w = np.expand_dims(state[10:], 1)
        f = np.expand_dims(control[:3], 1)
        tau = np.expand_dims(control[3:], 1)

        A_mat = A(p, v, q, w, f, tau)
        B_mat = B(p, v, q, w, f, tau)

        # Augmented system matrix
        Xi = np.zeros((A_mat.shape[0] * 3 + B_mat.shape[1], A_mat.shape[0] * 3 + B_mat.shape[1]))
        Xi[: A_mat.shape[0], : A_mat.shape[0]] = A_mat
        Xi[A_mat.shape[0] : 2 * A_mat.shape[0], A_mat.shape[0] : 2 * A_mat.shape[0]] = -A_mat.T
        Xi[2 * A_mat.shape[0] : 3 * A_mat.shape[0], 2 * A_mat.shape[0] : 3 * A_mat.shape[0]] = A_mat
        Xi[2 * A_mat.shape[0] : 3 * A_mat.shape[0], 3 * A_mat.shape[0] :] = B_mat

        # Matrix exponential for discretization
        Y = scipy.linalg.expm(Xi * dt)
        A_d = Y[: A_mat.shape[0], : A_mat.shape[0]]
        B_d = Y[2 * A_mat.shape[0] : 3 * A_mat.shape[0], 3 * A_mat.shape[0] :]
        return A_d, B_d

    def dVdt(self, t: float, V: np.ndarray, control_current: np.ndarray, control_next: np.ndarray, A, B, obstacles):
        """
        Time derivative of the augmented state vector for the system.

        Parameters are the same as outlined in the original function.
        """
        n_x = self.params["n_states"]
        n_u = self.params["n_controls"]

        # Define indices for slicing
        i0 = 0
        i1 = n_x
        i2 = i1 + n_x * n_x
        i3 = i2 + n_x * n_u
        i4 = i3 + n_x * n_u
        i5 = i4 + n_x

        x = V[i0:i1]

        # Interpolate control inputs
        if self.params["dis_type"] == "ZOH":
            beta = 0.0
        elif self.params["dis_type"] == "FOH":
            beta = t / self.params["dt_ss"]
        alpha = 1 - beta
        u = control_current + beta * (control_next - control_current)

        # Initialize Jacobians
        A_aug = np.zeros((n_x, n_x))
        B_aug = np.zeros((n_x, n_u))

        p, v, q, w, f, tau = (
            np.expand_dims(x[:3], 1),
            np.expand_dims(x[3:6], 1),
            np.expand_dims(x[6:10], 1),
            np.expand_dims(x[10 : self.params["n_states"] - self.params["n_obs"]], 1),
            np.expand_dims(u[:3], 1),
            np.expand_dims(u[3:], 1),
        )

        # Evaluate Jacobians
        A_subs = A(p, v, q, w, f, tau)
        A_aug[: self.params["n_states"] - self.params["n_obs"], : self.params["n_states"] - self.params["n_obs"]] = (
            A_subs
        )

        if self.params["ctcs"]:
            A_aug[self.params["n_states"] - len(obstacles) :, : len(p)] = np.array(
                [obs.grad_g_bar_ctcs(p) for obs in obstacles]
            )

        B_subs = B(p, v, q, w, f, tau)
        B_aug[: self.params["n_states"] - self.params["n_obs"]] = B_subs

        # Compute propagation term
        f_subs = self.augmented_dynamics(t, x, u, None, None, True, obstacles)

        z_t = np.squeeze(f_subs) - np.matmul(A_aug, x) - np.matmul(B_aug, u)

        # Stack results
        dVdt = np.zeros_like(V)
        dVdt[i0:i1] = f_subs.T
        dVdt[i1:i2] = np.matmul(A_aug, V[i1:i2].reshape((n_x, n_x))).reshape(-1)
        dVdt[i2:i3] = (np.matmul(A_aug, V[i2:i3].reshape((n_x, n_u))) + B_aug * alpha).reshape(-1)
        dVdt[i3:i4] = (np.matmul(A_aug, V[i3:i4].reshape((n_x, n_u))) + B_aug * beta).reshape(-1)
        dVdt[i4:i5] = np.matmul(A_aug, V[i4:i5]).reshape(-1) + z_t
        return dVdt

    def calculate_discretization(self, x: np.ndarray, u: np.ndarray, A, B, obstacles):
        """
        Calculate discretization for states and controls over time.

        Parameters are the same as outlined in the original function.
        """
        n_x = self.params["n_states"]
        n_u = self.params["n_controls"]

        i0 = 0
        i1 = n_x
        i2 = i1 + n_x * n_x
        i3 = i2 + n_x * n_u
        i4 = i3 + n_x * n_u
        i5 = i4 + n_x

        V0 = np.zeros(i5)
        V0[i1:i2] = np.eye(n_x).reshape(-1)

        f_bar = np.zeros((n_x, self.params["n"] - 1))
        A_bar = np.zeros((n_x * n_x, self.params["n"] - 1))
        B_bar = np.zeros((n_x * n_u, self.params["n"] - 1))
        C_bar = np.zeros((n_x * n_u, self.params["n"] - 1))
        z_bar = np.zeros((n_x, self.params["n"] - 1))

        for k in range(self.params["n"] - 1):
            V0[i0:i1] = x[:, k]

            V = solve_ivp(
                self.dVdt, (0, self.params["dt_ss"]), V0, args=(u[:, k], u[:, k + 1], A, B, obstacles), method="DOP853"
            ).y[:, -1]

            f_bar[:, k] = V[i0:i1]
            Phi = V[i1:i2].reshape((n_x, n_x))
            A_bar[:, k] = Phi.flatten(order="F")
            B_bar[:, k] = V[i2:i3].reshape((n_x, n_u)).flatten(order="F")
            C_bar[:, k] = V[i3:i4].reshape((n_x, n_u)).flatten(order="F")
            z_bar[:, k] = V[i4:i5]

        return A_bar, B_bar, C_bar, z_bar

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