"""探索方向の計算方法についてまとめたモジュール"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix as Csr

from ...logger import get_main_logger
from ...problem import LinearProgrammingProblemStandard as LPS
from ..linear_system_solver.exact_linear_system_solver import AbstractLinearSystemSolver
from ..variables import LPVariables

logger = get_main_logger()


class AbstractSearchDirectionCalculator(metaclass=ABCMeta):
    # 変数が変わっていなければ係数行列は変わらないため, 不要な計算を省くために前回の計算を記録しておく
    pre_x_divided_s: np.ndarray | None = None
    coef_matrix: Csr | None = None

    def __init__(self, linear_system_solver: AbstractLinearSystemSolver):
        self.linear_system_solver: AbstractLinearSystemSolver = linear_system_solver

    @abstractmethod
    def run(
        self, v: LPVariables, problem: LPS, right_hand_side: np.ndarray, tolerance: float | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """探索方向を解く.

        Args:
            v (LPVariables): その反復時点での変数
            problem (LPS): 問題
            right_hand_side (np.ndarray): 変形を行う前の探索方向の右辺ベクトル
            tolerance (float): 探索方向を計算する際の誤差許容度. ExactLinearSystemSolver なら不要

        Returns:
            np.ndarray: 解いた結果の探索方向 x
            np.ndarray: 解いた結果の探索方向 y
            np.ndarray: 解いた結果の探索方向 s
            float: 解いた際の結果のずれのベクトルのノルム
        """
        pass
