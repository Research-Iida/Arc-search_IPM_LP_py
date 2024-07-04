import copy
import time

import numpy as np

from ...logger import get_main_logger, indent
from ...problem import LinearProgrammingProblemStandard as LPS
from ...utils import config_utils
from ..solved_checker import InexactSolvedChecker, SolvedChecker
from ..solved_data import SolvedDetail
from ..variables import LPVariables
from .algorithm import ILPSolvingAlgoritm
from .inexact_interior_point_method import InexactArcSearchIPM, InexactLineSearchIPM
from .interior_point_method import ArcSearchIPM, InteriorPointMethod, LineSearchIPM

logger = get_main_logger()


class InnerSolverSelectionError(Exception):
    """solver の選択に失敗したときに発生するエラー"""

    pass


class IterativeRefinementSolvedChecker(SolvedChecker):
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
        if "delta_p_k" in kwargs or "delta_d_k" in kwargs:
            delta_p_k = kwargs["delta_p_k"]
            delta_d_k = kwargs["delta_d_k"]
            if (
                delta_p_k <= self.stop_criteria_threshold
                and delta_d_k <= self.stop_criteria_threshold
                and v.mu <= self.stop_criteria_threshold
            ):
                return True

        return self.is_relative_solved(v, problem)


class IterativeRefinementMethod(ILPSolvingAlgoritm):
    """Iterative Refinement によって問題を都度更新してLPを求解するクラス"""

    inner_algorithm: ILPSolvingAlgoritm

    @property
    def hat_zeta(self) -> float:
        return self.parameters.ITERATIVE_REFINEMENT_OPTIMAL_THRESHOLD_OF_SOLVER

    def _get_inner_algorithm(self) -> ILPSolvingAlgoritm:
        """Iterative Refinement 内部で実行する inexact solver の取得"""
        str_solver = self.parameters.ITERATIVE_REFINEMENT_INNER_SOLVER
        solved_checker = InexactSolvedChecker(self.hat_zeta, self.parameters.THRESHOLD_XS_NEGATIVE, False)
        match str_solver:
            case "inexact_arc":
                return InexactArcSearchIPM(self.config_section, solved_checker)
            case "inexact_line":
                return InexactLineSearchIPM(self.config_section, solved_checker)
            case "arc":
                return ArcSearchIPM(self.config_section, solved_checker)
            case "line":
                return LineSearchIPM(self.config_section, solved_checker)
            case _:
                raise InnerSolverSelectionError(f"Don't match solver for {str_solver}")

    def __init__(
        self,
        config_section: str = config_utils.default_section,
        stop_criteria_parameter: float | None = None,
        inner_algorithm: InteriorPointMethod | None = None,
    ):
        """初期化

        Args:
            config_section (str, optional): 使用するconfig. Defaults to config_utils.default_section.
            inner_algorithm (InteriorPointMethod | None, optional): 内部で使用するソルバー.
                インスタンス化されているので設定も含む. Defaults to None.
        """
        self._set_config_and_parameters(config_section)
        self.solved_checker = IterativeRefinementSolvedChecker(
            self.parameters.STOP_CRITERIA_PARAMETER, self.parameters.THRESHOLD_XS_NEGATIVE
        )

        if inner_algorithm is None:
            self.inner_algorithm = self._get_inner_algorithm()
        else:
            self.inner_algorithm = inner_algorithm
        logger.info(f"Inner solver is {self.inner_algorithm.__class__.__name__}.")

    @property
    def zeta(self) -> float:
        return self.parameters.STOP_CRITERIA_PARAMETER

    @property
    def rho(self) -> float:
        return self.parameters.ITERATIVE_REFINEMENT_SCALING_MULTIPLIER

    def run_inner_algorithm(self, problem: LPS, v_0: LPVariables | None = None) -> SolvedDetail:
        """内部で設定したアルゴリズムに解かせる.

        Args:
            problem (LPS): 対象とするLP
            v_0 (LPVariables | None): 初期点. 1回目以降の refinement では初期点は `inner_algorithm` に作成を任せた方がよいため,
                None になる

        Returns:
            SolvedDetail: 解
        """
        # ログで見やすいように線を入れておく
        logger.info(f"{'-' * 3} inner solver start {'-' * 50}")
        result = self.inner_algorithm.run(problem, v_0)
        logger.info(f"{'-' * 3} inner solver end {'-' * 50}")
        return result

    def run(self, problem_0: LPS, v_0: LPVariables | None) -> SolvedDetail:
        """反復で解くアルゴリズム部分の実行

        Returns:
            SolvedDetail: 最適解に関する詳細情報を格納したデータ構造
        """
        # 実行時間記録開始
        start_time = time.time()

        problem_name_prefix = f"{problem_0.name}_refined_"
        count_iterative_refinement = 0
        large_delta_p_k = 1
        large_delta_d_k = 1

        # 最初の求解
        aSolvedDetail = self.run_inner_algorithm(problem_0, v_0)
        v_star = aSolvedDetail.v
        # problem = aSolvedDetail.problem
        iter_num = aSolvedDetail.aSolvedSummary.iter_num

        b_bar = problem_0.b - problem_0.A @ v_star.x
        c_bar = problem_0.c - problem_0.A.T @ v_star.y

        delta_p_k = max(np.abs(b_bar))
        delta_d_k = max(max(-c_bar), 0)
        logger.info(f"delta_p_k: {delta_p_k}, delta_d_k: {delta_d_k}")

        is_terminated = self.is_terminate(
            v_star,
            problem_0,
            count_iterative_refinement,
            time.time() - start_time,
            delta_p_k,
            delta_d_k,
            aSolvedDetail,
        )

        lst_variables = copy.deepcopy(aSolvedDetail.lst_variables_by_iter)
        lst_mu = copy.deepcopy(aSolvedDetail.lst_mu_by_iter)
        lst_alpha_x = copy.deepcopy(aSolvedDetail.lst_main_step_size_by_iter)
        lst_alpha_s = copy.deepcopy(aSolvedDetail.lst_dual_step_size_by_iter)
        lst_norm_vdot = copy.deepcopy(aSolvedDetail.lst_norm_vdot_by_iter)
        lst_norm_vddot = copy.deepcopy(aSolvedDetail.lst_norm_vddot_by_iter)
        lst_max_norm_main_const = copy.deepcopy(aSolvedDetail.lst_max_norm_main_constraint_by_iter)
        lst_max_norm_dual_const = copy.deepcopy(aSolvedDetail.lst_max_norm_dual_constraint_by_iter)
        lst_residual_inexact_vdot = copy.deepcopy(aSolvedDetail.lst_residual_inexact_vdot)
        lst_tolerance_inexact_vdot = copy.deepcopy(aSolvedDetail.lst_tolerance_inexact_vdot)
        lst_residual_inexact_vddot = copy.deepcopy(aSolvedDetail.lst_residual_inexact_vddot)
        lst_tolerance_inexact_vddot = copy.deepcopy(aSolvedDetail.lst_tolerance_inexact_vddot)
        lst_iteration_number_updated_by_iterative_refinement: list[float] = []

        # 収束条件を満たすまで反復する
        while not is_terminated:
            count_iterative_refinement += 1
            logger.info(f"Iterative Refinement number: {count_iterative_refinement}")
            logger.info(f"{indent}total iteration: {iter_num}")

            # 定数の更新
            # QIPM の論文に合わせる場合
            # large_delta_p_k = 2 ** np.ceil(np.log(1 / max(delta_p_k, 1 / (self.rho * large_delta_p_k))))
            # large_delta_d_k = 2 ** np.ceil(np.log(1 / max(delta_d_k, 1 / (self.rho * large_delta_d_k))))
            # iterative refinement の元論文に合わせる場合
            large_delta_p_k = 1 / max(delta_p_k, 1 / (self.rho * large_delta_p_k))
            large_delta_d_k = 1 / max(delta_d_k, 1 / (self.rho * large_delta_d_k))
            logger.info(f"large_delta_p_k: {large_delta_p_k}, large_delta_d_k: {large_delta_d_k}")

            # 問題の更新
            problem = LPS(
                problem_0.A,
                large_delta_p_k * (b_bar + problem_0.A @ v_star.x),
                large_delta_d_k * c_bar,
                f"{problem_name_prefix}{count_iterative_refinement}",
            )

            # inexact ソルバーで求解
            aSolvedDetail = self.run_inner_algorithm(problem)
            v_hat = aSolvedDetail.v
            # problem = aSolvedDetail.problem
            iter_num += aSolvedDetail.aSolvedSummary.iter_num

            # 記録と更新
            x_star = v_hat.x / large_delta_p_k
            y_star = v_star.y + v_hat.y / large_delta_d_k

            b_bar = problem_0.b - problem_0.A @ x_star
            c_bar = problem_0.c - problem_0.A.T @ y_star

            delta_p_k = max(np.abs(b_bar))
            delta_d_k = max(max(-c_bar), 0)
            logger.info(f"delta_p_k: {delta_p_k}, delta_d_k: {delta_d_k}")

            # s は最適解に到達しているかの判定にのみ使用する
            # 理論的には c - A^T y >= -epsilon になるはず, 誤差で負の値になるため0以上に修正
            s_star = np.where(c_bar < 0, 0, c_bar)
            v_star = LPVariables(x_star, y_star, s_star)

            lst_variables += aSolvedDetail.lst_variables_by_iter
            lst_mu += aSolvedDetail.lst_mu_by_iter
            lst_alpha_x += aSolvedDetail.lst_main_step_size_by_iter
            lst_alpha_s += aSolvedDetail.lst_dual_step_size_by_iter
            lst_norm_vdot += aSolvedDetail.lst_norm_vdot_by_iter
            lst_norm_vddot += aSolvedDetail.lst_norm_vddot_by_iter
            lst_max_norm_main_const += aSolvedDetail.lst_max_norm_main_constraint_by_iter
            lst_max_norm_dual_const += aSolvedDetail.lst_max_norm_dual_constraint_by_iter
            lst_residual_inexact_vdot += aSolvedDetail.lst_residual_inexact_vdot
            lst_tolerance_inexact_vdot += aSolvedDetail.lst_tolerance_inexact_vdot
            lst_residual_inexact_vddot += aSolvedDetail.lst_residual_inexact_vddot
            lst_tolerance_inexact_vddot += aSolvedDetail.lst_tolerance_inexact_vddot
            lst_iteration_number_updated_by_iterative_refinement.append(iter_num)

            logger.info(f"Duality parameter: {v_star.mu}")
            logger.info("Max constraint violation:")
            logger.info(f"{indent}main: {max(np.abs(problem_0.residual_main_constraint(v_star.x)))}")
            logger.info(f"{indent}dual: {max(np.abs(problem_0.residual_dual_constraint(v_star.y, v_star.s)))}")

            is_terminated = self.is_terminate(
                v_star,
                problem_0,
                count_iterative_refinement,
                time.time() - start_time,
                delta_p_k,
                delta_d_k,
                aSolvedDetail,
            )

        # 時間計測終了
        elapsed_time = time.time() - start_time
        logger.info(f"Needed iterative refinement number: {count_iterative_refinement}")

        # 出力の作成
        # delta_k での判定は iterative refinement 特有のもので問題自体を解けたかは不明. なので relative な判定を行う
        is_solved = self.solved_checker.is_relative_solved(v_star, problem_0)
        # 反復回数上限に達したかどうかは iterative refinement を何回行ったかで判断
        aSolvedSummary = self.make_SolvedSummary(
            v_star,
            problem_0,
            is_solved,
            iter_num,
            self.is_iteration_number_reached_upper(count_iterative_refinement, problem_0),
            elapsed_time,
        )
        output = SolvedDetail(
            aSolvedSummary,
            v_star,
            problem_0,
            v_0,
            problem_0,
            lst_variables_by_iter=lst_variables,
            lst_main_step_size_by_iter=lst_alpha_x,
            lst_dual_step_size_by_iter=lst_alpha_s,
            lst_mu_by_iter=lst_mu,
            lst_norm_vdot_by_iter=lst_norm_vdot,
            lst_norm_vddot_by_iter=lst_norm_vddot,
            lst_max_norm_main_constraint_by_iter=lst_max_norm_main_const,
            lst_max_norm_dual_constraint_by_iter=lst_max_norm_dual_const,
            lst_iteration_number_updated_by_iterative_refinement=lst_iteration_number_updated_by_iterative_refinement,
            lst_residual_inexact_vdot=lst_residual_inexact_vdot,
            lst_tolerance_inexact_vdot=lst_tolerance_inexact_vdot,
            lst_residual_inexact_vddot=lst_residual_inexact_vddot,
            lst_tolerance_inexact_vddot=lst_tolerance_inexact_vddot,
        )
        return output

    def is_iteration_number_reached_upper(self, iter_num: int, problem: LPS) -> bool:
        """反復回数が上限を超えたか

        Iterative refinement の反復回数上限に達したかで判定
        本来は O(L) だが, L を算出する意味があまり見いだせない

        Args:
            iter_num: iterative refinement 実施回数
        """
        return iter_num >= self.parameters.ITERATIVE_REFINEMENT_ITER_UPPER

    def is_terminate(
        self,
        v: LPVariables,
        problem: LPS,
        count_iterative_refinement: int,
        elapsed_time: float,
        delta_p_k: float,
        delta_d_k: float,
        inner_solved_data: SolvedDetail,
    ) -> bool:
        if self.solved_checker.run(v, problem, delta_p_k=delta_p_k, delta_d_k=delta_d_k):
            logger.info(f"{indent}Variables satisfy the solution condition.")
            return True
        # 変数が最適性を満たしていれば, iterative refinement の終了条件に関わらず終了する
        if self.solved_checker.is_relative_solved(v, problem):
            logger.info(f"{indent}Variables satisfy the solution condition with relative criteria.")
            return True

        if self.is_iteration_number_reached_upper(count_iterative_refinement, problem):
            logger.warning(f"{indent}Algorithm terminates upper bound of iterative refinement.")
            return True

        if self.is_calculation_time_reached_upper(elapsed_time):
            logger.warning(f"{indent}Algorithm terminates calculation time upper bound.")
            return True

        # inner solver が求解できなかった場合は終了
        if not inner_solved_data.aSolvedSummary.is_solved:
            logger.warning(f"{indent}Inner solver cannot solve {inner_solved_data.aSolvedSummary.problem_name}.")
            return True

        return False
