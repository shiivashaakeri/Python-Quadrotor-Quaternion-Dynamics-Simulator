import numpy as np
from sympy import Matrix, MatrixSymbol, lambdify

from params import rocket_params
from utils.quaternion_utils import QuaternionUtils


class Linearization:
    def __init__(self, params):
        """
        Initialize the Linearization class with system parameters.

        Parameters:
        params (dict): Parameters of the system, such as mass ('m'), gravity ('g'), and inertia tensor ('J_b').
        """
        self.params = params

    def analytical_jacobians(self):
        """
        Computes the analytical Jacobians of the system dynamics.

        Returns:
        A, B: Functions that compute the Jacobians of the system dynamics with respect to the state and the control input.
        """
        # Define symbols for the state and control input
        r = MatrixSymbol("r", 3, 1)  # Position
        v = MatrixSymbol("v", 3, 1)  # Velocity
        q = MatrixSymbol("q", 4, 1)  # Attitude (Quaternion)
        w = MatrixSymbol("w", 3, 1)  # Angular Rate
        f = MatrixSymbol("f", 3, 1)  # Applied Force
        tau = MatrixSymbol("tau", 3, 1)  # Applied Torque

        # Compute the time derivatives of the state variables
        r_dot = v
        v_dot = (1 / self.params["m"]) * QuaternionUtils.qdcm(q) @ f + Matrix([0, 0, self.params["g"]])
        q_dot = 0.5 * QuaternionUtils.skew_symmetric_matrix_quat(w) @ q
        w_dot = np.diag(1 / self.params["J_b"]) @ (
            tau - QuaternionUtils.skew_symmetric_matrix(w) @ np.diag(self.params["J_b"]) @ w
        )

        # Combine the time derivatives of the state variables into a single matrix
        state_dot = Matrix.vstack(Matrix(r_dot), Matrix(v_dot), Matrix(q_dot), Matrix(w_dot))

        # Compute the Jacobians of the system dynamics with respect to the state and the control input
        A_expr = state_dot.jacobian(Matrix([r, v, q, w]))
        B_expr = state_dot.jacobian(Matrix([f, tau]))

        # Convert the Jacobians to functions that can be evaluated with numerical values
        A = lambdify((r, v, q, w, f, tau), A_expr, "numpy")
        B = lambdify((r, v, q, w, f, tau), B_expr, "numpy")

        return A, B
