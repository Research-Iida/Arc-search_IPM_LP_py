"""探索方向の計算方法についてまとめたモジュール"""
from abc import ABCMeta, abstractmethod

import numpy as np

from ..logger import get_main_logger
from ..problem import LinearProgrammingProblemStandard as LPS
from ..linear_system_solver.exact_linear_system_solver import AbstractLinearSystemSolver
from .variables import LPVariables

logger = get_main_logger()


class SelectionBasisError(Exception):
    pass


class AbstractSearchDirectionCalculator(metaclass=ABCMeta):
    # 変数が変わっていなければ係数行列は変わらないため, 不要な計算を省くために前回の計算を記録しておく
    pre_x_divided_s: np.ndarray | None = None
    coef_matrix: np.ndarray | None = None

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


class NESSearchDirectionCalculator(AbstractSearchDirectionCalculator):
    def run(
        self, v: LPVariables, problem: LPS, right_hand_side: np.ndarray,
        tolerance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """NES の定式化で探索方向を解く.
        IPMの基本はこちら

        Note:
            * iida の inexact arc-search の論文では MNES の定式化を使用するが, QIPM の実装では NES でやっている

        Args:
            v (LPVariables): その反復時点での変数
            problem (LPS): 問題
            right_hand_side (np.ndarray): 変形を行う前の探索方向の右辺ベクトル
            tolerance (float): 探索方向を計算する際の誤差許容度. ExactLinearSystemSolver なら不要

        Returns:
            np.ndarray: NES で解いた結果の探索方向 x
            np.ndarray: NES で解いた結果の探索方向 y
            np.ndarray: NES で解いた結果の探索方向 s
            float: NES で解いた際の結果のずれのベクトルのノルム
        """
        m = problem.m
        n = problem.n
        A = problem.A
        x_divided_s = v.x / v.s
        AXS_inv = A @ np.diag(x_divided_s)

        # NES に変形
        if self.pre_x_divided_s is None or np.any(self.pre_x_divided_s != x_divided_s):
            logger.info("Update coefficient matrix.")
            coef_matrix = AXS_inv @ A.T
            # logger.debug(f"{indent}NES coef matrix condition number: {np.linalg.cond(coef_matrix)}")

            self.pre_x_divided_s = x_divided_s
            self.coef_matrix = coef_matrix

        right_hand_side_NES = AXS_inv @ right_hand_side[m:n + m] + right_hand_side[:m] - A @ (right_hand_side[n + m:] / v.s)

        # inexact に求解
        sol_NES = self.linear_system_solver.solve(self.coef_matrix, right_hand_side_NES, tolerance=tolerance)
        residual_sol = self.coef_matrix @ sol_NES - right_hand_side_NES

        # 解から復元
        sol_y = sol_NES
        sol_s = right_hand_side[m:n + m] - A.T @ sol_y
        sol_x = -(v.x * sol_s / v.s) + right_hand_side[n + m:] / v.s
        return sol_x, sol_y, sol_s, np.linalg.norm(residual_sol)
