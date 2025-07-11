"""ソルバーが問題を解くことができたかを確認するクラス"""

from abc import ABCMeta, abstractmethod

import numpy as np

from ..logger import get_main_logger, indent
from ..problem import LinearProgrammingProblemStandard as LPS
from .variables import LPVariables

logger = get_main_logger()


class SolvedChecker(metaclass=ABCMeta):
    """_summary_

    Attributes:
        stop_criteria_threshold: どれだけ最適解に肉薄していればアルゴリズムを停止するかの threshold.
            ソルバーによって停止条件は異なるため, 使われ方はソルバーごとに異なる
    """

    stop_criteria_threshold: float
    threshold_xs_negative: float

    def __init__(self, stop_criteria_threshold: float, threshold_xs_negative: float):
        self.stop_criteria_threshold = stop_criteria_threshold
        self.threshold_xs_negative = threshold_xs_negative

    def is_xs_positive(self, v: LPVariables) -> bool:
        """x, s が positive であることを確認する

        Args:
            v (LPVariables): 変数 x,s が格納されたインスタンス

        Returns:
            bool: x,s がともに strictly positive ならば True
        """
        result = True
        if len(v.x[v.x < -self.threshold_xs_negative]):
            logger.warning("Solution x is negative. Cannot solve this problem!")
            idx_ = np.where(v.x < -self.threshold_xs_negative)
            logger.warning(f"Negative index: {idx_}, Value: {v.x[v.x < -self.threshold_xs_negative]}")
            result = False
        if len(v.s[v.s < -self.threshold_xs_negative]):
            logger.warning("Solution s is negative. Cannot solve this problem!")
            idx_ = np.where(v.s < -self.threshold_xs_negative)
            logger.warning(f"Negative index: {idx_}, Value: {v.s[v.s < -self.threshold_xs_negative]}")
            result = False
        return result

    @abstractmethod
    def run(
        self,
        v: LPVariables,
        problem: LPS,
        *args,
        **kwargs,
    ) -> bool:
        """アルゴリズムが最適性を満たし, 最適解にたどり着いたかを確認
        アルゴリズムごとに求解条件が異なるため, 実装は具象クラスへ移譲
        """
        pass


class RelativeSolvedChecker(SolvedChecker):
    """Merotra の手法で確認する"""

    def run(
        self,
        v: LPVariables,
        problem: LPS,
        *args,
        **kwargs,
    ) -> bool:
        """Merotra の手法である, relative な終了条件で確認する
        参考: ``Arc-Search Techniques for Interior-Point Methods'' Section 7.3.10
        """
        # x, s のどちらかが0未満だった場合実行不可能となる
        if not self.is_xs_positive(v):
            return False

        r_b = problem.residual_main_constraint(v.x)
        r_c = problem.residual_dual_constraint(v.y, v.s)
        criteria_main_resi = np.linalg.norm(r_b) / max(1, np.linalg.norm(problem.b))
        criteria_dual_resi = np.linalg.norm(r_c) / max(1, np.linalg.norm(problem.c))
        logger.debug(f"{criteria_main_resi=}, {criteria_dual_resi=}")

        obj_main_norm = np.linalg.norm(problem.objective_main(v.x))
        obj_dual_norm = np.linalg.norm(problem.objective_dual(v.y))
        criteria_duality = v.mu / max(1, obj_main_norm, obj_dual_norm)
        logger.debug(f"{criteria_duality=}")
        criteria = max(criteria_main_resi, criteria_dual_resi, criteria_duality)
        logger.debug(f"{criteria=}, {self.stop_criteria_threshold=}")
        return criteria < self.stop_criteria_threshold


class ArcIPMWithRestartingProvenSolvedChecker(SolvedChecker):
    """
    ``An infeasible interior-point arc-search method with Nesterov's restarting strategy for linear programming problems''
    での停止条件で確認する"""

    relative_checker: RelativeSolvedChecker

    def __init__(self, stop_criteria_threshold: float, threshold_xs_negative: float):
        super().__init__(stop_criteria_threshold, threshold_xs_negative)
        self.relative_checker = RelativeSolvedChecker(stop_criteria_threshold, threshold_xs_negative)

    def run(
        self,
        v: LPVariables,
        problem: LPS,
        *args,
        **kwargs,
    ) -> bool:
        """アルゴリズムが最適性を満たし, 最適解にたどり着いたかを確認"""
        if "mu_0" not in kwargs or "r_b_0" not in kwargs or "r_c_0" not in kwargs:
            logger.info("Input doesn't have mu_0, r_b_0 and r_c_0. We use relative solved checker.")
            return self.relative_checker.run(v, problem)
        mu_0 = kwargs["mu_0"]

        # x, s のどちらかが0未満だった場合実行不可能となる
        if not self.is_xs_positive(v):
            return False

        is_small_mu = v.mu < self.stop_criteria_threshold
        r_b = problem.residual_main_constraint(v.x)
        is_small_residual_main = (
            np.linalg.norm(r_b) < self.stop_criteria_threshold * np.linalg.norm(kwargs["r_b_0"]) / mu_0
        )
        r_c = problem.residual_dual_constraint(v.y, v.s)
        is_small_residual_dual = (
            np.linalg.norm(r_c) < self.stop_criteria_threshold * np.linalg.norm(kwargs["r_c_0"]) / mu_0
        )
        return is_small_mu and is_small_residual_main and is_small_residual_dual


class AbsoluteSolvedChecker(SolvedChecker):
    """mu < epsilon, r_b, r_c < epsilon であることを確認するクラス"""

    def run(
        self,
        v: LPVariables,
        problem: LPS,
        *args,
        **kwargs,
    ) -> bool:
        """アルゴリズムが最適性を満たし, 最適解にたどり着いたか"""
        # x, s のどちらかが0未満だった場合実行不可能となる
        if not self.is_xs_positive(v):
            return False

        is_small_mu = v.mu <= self.stop_criteria_threshold
        r_b = problem.residual_main_constraint(v.x)
        r_c = problem.residual_dual_constraint(v.y, v.s)
        is_small_residual = np.linalg.norm(np.concatenate([r_b, r_c]), ord=np.inf) <= self.stop_criteria_threshold
        return is_small_mu and is_small_residual


class IterativeRefinementSolvedChecker(SolvedChecker):
    relative_checker: RelativeSolvedChecker

    def __init__(self, stop_criteria_threshold: float, threshold_xs_negative: float):
        super().__init__(stop_criteria_threshold, threshold_xs_negative)
        self.relative_checker = RelativeSolvedChecker(stop_criteria_threshold, threshold_xs_negative)

    def run(
        self,
        v: LPVariables,
        problem: LPS,
        *args,
        **kwargs,
    ) -> bool:
        """Iterative Refinement が最適性を満たし, 最適解にたどり着いたかを確認

        アルゴリズム実行中は delta_k が存在するので, relative に加えてそちらでも判定を行う.
        もしなければ（LPSolver で SolvedSummary 作るときとか）は relative のみで判定
        """
        # x, s のどちらかが0未満だった場合実行不可能となる
        if not self.is_xs_positive(v):
            return False

        if "delta_p_k" in kwargs or "delta_d_k" in kwargs:
            delta_p_k = kwargs["delta_p_k"]
            delta_d_k = kwargs["delta_d_k"]
            if (
                delta_p_k <= self.stop_criteria_threshold
                and delta_d_k <= self.stop_criteria_threshold
                and v.mu <= self.stop_criteria_threshold
            ):
                return True

        logger.info(f"{indent}Check variables satisfying the solution condition with relative criteria.")
        return self.relative_checker.run(v, problem)
