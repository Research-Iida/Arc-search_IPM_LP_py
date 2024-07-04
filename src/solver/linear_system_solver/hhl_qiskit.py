import numpy as np
from qiskit.algorithms.linear_solvers.hhl import HHL
from qiskit.quantum_info import Statevector

from ...logger import get_main_logger
from .inexact_linear_system_solver import AbstractInexactLinearSystemSolver

logger = get_main_logger()


class HHLLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """HHL アルゴリズムによる求解"""

    def solve(self, A: np.ndarray, b: np.ndarray, tolerance: float = 10**-7, *args) -> np.ndarray:
        # 係数行列の行数は2のべき乗でなければならないらしい. 拡張
        m = A.shape[0]
        state_vector_digits = int(np.log2(m))
        dim_for_hhl = pow(2, state_vector_digits + 1)
        A_modified = np.eye(dim_for_hhl)
        A_modified[0:m, 0:m] = A.copy()
        b_modified = np.zeros(dim_for_hhl)
        b_modified[0:m] = b.copy()

        # 右辺は単位ベクトルに修正
        normalize_coef = np.linalg.norm(b_modified)
        A_normalized = A_modified / normalize_coef
        b_normalized = b_modified / normalize_coef

        sol_hhl = HHL(epsilon=tolerance).solve(A_normalized, b_normalized)

        # 解の復元
        naive_sv = Statevector(sol_hhl.state).data
        start_index = pow(2, state_vector_digits - 1)  # remove ancilla bit
        naive_full_vector = np.array(naive_sv[start_index : start_index + dim_for_hhl])

        sol_vector = sol_hhl.euclidean_norm * np.real(naive_full_vector) / np.linalg.norm(naive_full_vector)
        return sol_vector[0:m]
