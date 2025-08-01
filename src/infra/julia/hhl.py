import numpy as np
from julia import Main

from ...logger import get_main_logger
from ...solver.linear_system_solver.inexact_linear_system_solver import AbstractInexactLinearSystemSolver

logger = get_main_logger()


class HHLJuliaLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """HHL アルゴリズムによる求解"""

    def __init__(self, n_pe_qubits: int):
        """インスタンス初期化

        Args:
            n_pe_qubits (int): Phase Estimator に使用する qubit 数.
                Aの固有値の精度に関わる
        """
        Main.include("src/solver/linear_system_solver/HHLlib.jl")

        self.n_pe_qubits = n_pe_qubits

    def solve(self, A: np.ndarray, b: np.ndarray, tolerance: float = 10**-7, *args) -> np.ndarray:
        # 数値誤差により係数行列が対称にならない場合があるため, 少し修正
        coef_matrix = (A + A.T) / 2
        right_hand_side = b.copy()

        # A is positive definite, hermitian, with its maximum eigenvalue λ_max < 1.
        eig_vals, _ = np.linalg.eig(coef_matrix)
        max_eigen_value = max(eig_vals)
        if max_eigen_value >= 1:
            coef_eig_val_normalize = max_eigen_value + 10 ** (-3)
            coef_matrix /= coef_eig_val_normalize
        else:
            coef_eig_val_normalize = 1

        # b is normalized.
        coef_rhs_normalize = np.linalg.norm(right_hand_side)
        right_hand_side /= coef_rhs_normalize

        # 係数行列の行数は2のべき乗でなければならないらしい. 拡張
        m = coef_matrix.shape[0]
        log_m = np.log2(m)
        state_vector_digits = int(log_m)
        if log_m == state_vector_digits:
            dim_for_hhl = m
        else:
            dim_for_hhl = pow(2, state_vector_digits + 1)
        coef_matrix_modified = np.eye(dim_for_hhl)
        coef_matrix_modified[0:m, 0:m] = coef_matrix
        rhs_modified = np.zeros(dim_for_hhl)
        rhs_modified[0:m] = right_hand_side

        sol_hhl = Main.hhlsolve(coef_matrix_modified, rhs_modified, self.n_pe_qubits)
        return sol_hhl[0:m].real * coef_rhs_normalize / coef_eig_val_normalize
