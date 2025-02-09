import numpy as np


class QuaternionUtils:
    @staticmethod
    def qdcm(q: np.ndarray) -> np.ndarray:
        q_norm = (q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2)**0.5
        w, x, y, z = q / q_norm
        return np.array(
            [
                [1 - 2 * (y**2 + z**2), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x**2 + z**2), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)],
            ]
        )

    @staticmethod
    def skew_symmetric_matrix_quat(w: np.ndarray) -> np.ndarray:
        x, y, z = w
        return np.array([[0, -x, -y, -z], [x, 0, z, -y], [y, -z, 0, x], [z, y, -x, 0]])

    @staticmethod
    def skew_symmetric_matrix(w: np.ndarray) -> np.ndarray:
        x, y, z = w
        return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
