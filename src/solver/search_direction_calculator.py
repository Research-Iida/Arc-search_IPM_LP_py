"""探索方向の計算方法についてまとめたモジュール"""

import itertools
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.sparse import csr_matrix as Csr

from ..linear_system_solver.exact_linear_system_solver import AbstractLinearSystemSolver
from ..logger import get_main_logger, indent
from ..problem import LinearProgrammingProblemStandard as LPS
from .variables import LPVariables

logger = get_main_logger()


class SelectionBasisError(Exception):
    pass


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


class NESSearchDirectionCalculator(AbstractSearchDirectionCalculator):
    def run(
        self,
        v: LPVariables,
        problem: LPS,
        right_hand_side: np.ndarray,
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
        A: Csr = problem.A
        x_divided_s = v.x / v.s
        AXS_inv = A @ np.diag(x_divided_s)

        # NES に変形
        if self.pre_x_divided_s is None or np.any(self.pre_x_divided_s != x_divided_s):
            logger.info("Update coefficient matrix.")
            coef_matrix = AXS_inv @ A.T
            # logger.debug(f"{indent}NES coef matrix condition number: {np.linalg.cond(coef_matrix)}")

            self.pre_x_divided_s = x_divided_s
            self.coef_matrix = coef_matrix

        right_hand_side_NES = (
            AXS_inv @ right_hand_side[m : n + m] + right_hand_side[:m] - A @ (right_hand_side[n + m :] / v.s)
        )

        # inexact に求解
        sol_NES = self.linear_system_solver.solve(self.coef_matrix, right_hand_side_NES, tolerance=tolerance)
        residual_sol = self.coef_matrix @ sol_NES - right_hand_side_NES

        # 解から復元
        sol_y = sol_NES
        sol_s = right_hand_side[m : n + m] - A.T @ sol_y
        sol_x = -(v.x * sol_s / v.s) + right_hand_side[n + m :] / v.s
        return sol_x, sol_y, sol_s, np.linalg.norm(residual_sol)


class MNESSearchDirectionCalculator(AbstractSearchDirectionCalculator):
    # 初期点時点で決定できるものや1回計算すればいいものは Attributes として使いまわす
    A_base_indexes: list[int] | None = None

    def select_base_indexes(self, problem: LPS) -> list[int]:
        """MNES, OSS 定式化で使用する基底の index を選択
        先頭から1つずつ列を選んで, rank が上がれば基底として加える.

        TODO:
            * 現在のやり方だと full rank にならず永遠に回り続ける場合がある

        Args:
            problem (LPS): 選択対象の問題

        Returns:
            list[int]: index
        """
        # すでに計算しているならそこから取る
        if self.A_base_indexes is not None:
            return self.A_base_indexes

        doing_msg = "selecting A base"
        logger.info(f"Start {doing_msg}.")
        logger.debug(f"rank(A): {np.linalg.matrix_rank(problem.A)}, m: {problem.m}")

        base_idxs: set[int] = set()
        # 行で非ゼロ要素が2つ以下しかない列は線形独立としてよい
        rows_nonzero, columns_nonzero = np.where(np.abs(problem.A) > 10 ** (-6))
        for i in range(problem.m):
            idxs_row_nonzero = np.where(rows_nonzero == i)
            if len(idxs_row_nonzero[0]) <= 2:
                base_idxs.add(columns_nonzero[idxs_row_nonzero[0][0]])
        logger.info(f"Number of columns in only one or two nonzero element row: {len(base_idxs)}")

        # ``Convergence analysis of a long-step primal-dual infeasible interior-point LP algorithm based on iterative linear solvers''
        cycle_number = 0
        added_column_number_in_cycle = 0
        for add_column_number in itertools.cycle(range(problem.n)):
            if len(base_idxs) == problem.m:
                break

            if add_column_number in base_idxs:
                continue
            logger.debug(f"{indent}Target column: {add_column_number}, selected base number: {len(base_idxs)}")

            # なんべんも計算するので変数化
            if np.linalg.matrix_rank(problem.A[:, list(base_idxs) + [add_column_number]]) >= len(base_idxs) + 1:
                base_idxs.add(add_column_number)
                added_column_number_in_cycle += 1

            if add_column_number == problem.n - 1:
                logger.info(f"End cycle number: {cycle_number}, selected base number: {len(base_idxs)}")
                if added_column_number_in_cycle == 0:
                    raise SelectionBasisError("Cannot select basis in this cycle!")
                cycle_number += 1
                added_column_number_in_cycle = 0

        # 計算誤差程度の大きさは rank を計算する際に rank 落ちになる可能性があるため排除
        # indexes_A_nonzero = np.where(np.abs(problem.A) > 10**(-4))

        # 対角成分が非ゼロの行はどの列にそれがあるか確認
        # for i in range(problem.m):
        #     lst_nonzero_column = indexes_A_nonzero[1][np.where(indexes_A_nonzero[0] == i)]
        #     if i not in lst_nonzero_column:
        #         logger.debug(f"{i}-th row doesn't have {i}-th column: {lst_nonzero_column}, {problem.A[i, lst_nonzero_column]}")

        # 上三角行列を仮定し, 新しく非ゼロが出た列を選択
        # for i, j in zip(indexes_A_nonzero[0], indexes_A_nonzero[1]):
        #     if i < len(base_idxs):
        #         continue
        #     if j not in base_idxs:
        #         # 選択する内容の確認
        #         logger.debug(f"Selected row: {i}, column: {j}, element: {problem.A[i, j]}")
        #         base_idxs.append(j)

        # 出力の確認
        logger.debug(f"Selected base columns: {base_idxs}")
        rank_selected_A = np.linalg.matrix_rank(problem.A[:, list(base_idxs)])
        logger.debug(f"{indent}selected base num: {len(base_idxs)}, m: {problem.m}, rank: {rank_selected_A}")
        if rank_selected_A < problem.m:
            logger.warning(f"{indent}Selected base is not full rank!")
            assert False
        logger.info(f"End {doing_msg}.")

        # 一回計算したら記録しておく
        self.A_base_indexes = list(base_idxs)
        return self.A_base_indexes

    def run(
        self,
        v: LPVariables,
        problem: LPS,
        right_hand_side: np.ndarray,
        tolerance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """MNES の定式化で探索方向を解く.

        Args:
            v (LPVariables): その反復時点での変数
            problem (LPS): 問題
            right_hand_side (np.ndarray): 変形を行う前の探索方向の右辺ベクトル
            tolerance (float): 探索方向を計算する際の誤差許容度. ExactLinearSystemSolver なら不要

        Returns:
            np.ndarray: MNES で解いた結果の探索方向 x
            np.ndarray: MNES で解いた結果の探索方向 y
            np.ndarray: MNES で解いた結果の探索方向 s
            float: MNES で解いた際の結果のずれのベクトルのノルム
        """
        m = problem.m
        n = problem.n
        A: Csr = problem.A
        x_divided_s = v.x / v.s
        AXS_inv = A @ np.diag(x_divided_s)

        # MNES に変形
        hat_B = self.select_base_indexes(problem)
        D_B = np.diag(np.sqrt(x_divided_s[hat_B]))
        D_B_inv_A_B_inv = np.linalg.inv(D_B) @ np.linalg.inv(A[:, hat_B])

        if self.pre_x_divided_s is None or np.any(self.pre_x_divided_s != x_divided_s):
            logger.info("Update coefficient matrix.")
            # 事前に NES での係数行列を作ると A.T をかけることで数値誤差が出てきてしまうらしい. なのでここで一気に作成
            coef_matrix: Csr = D_B_inv_A_B_inv @ AXS_inv @ A.T @ D_B_inv_A_B_inv.T
            logger.info(f"{indent}MNES coef matrix condition number: {np.linalg.cond(coef_matrix.toarray())}")

            self.pre_x_divided_s = x_divided_s
            self.coef_matrix = coef_matrix

        right_hand_side_NES = (
            AXS_inv @ right_hand_side[m : n + m] + right_hand_side[:m] - A @ (right_hand_side[n + m :] / v.s)
        )
        right_hand_side_MNES = D_B_inv_A_B_inv @ right_hand_side_NES

        # inexact に求解
        sol_MNES = self.linear_system_solver.solve(self.coef_matrix, right_hand_side_MNES, tolerance=tolerance)
        residual_sol = self.coef_matrix @ sol_MNES - right_hand_side_MNES

        # 解から復元
        sol_y = D_B_inv_A_B_inv.T @ sol_MNES
        sol_s = right_hand_side[m : n + m] - A.T @ sol_y
        v_k = np.zeros(problem.n)
        v_k[hat_B] = D_B @ residual_sol
        sol_x = -(v.x * sol_s / v.s) + right_hand_side[n + m :] / v.s - v_k
        return sol_x, sol_y, sol_s, np.linalg.norm(residual_sol)
