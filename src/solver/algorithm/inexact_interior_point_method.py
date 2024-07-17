import time
from abc import ABCMeta

import numpy as np

from ...logger import get_main_logger, indent
from ...problem import LinearProgrammingProblemStandard as LPS
from ..optimization_parameters import OptimizationParameters
from ..solved_checker import SolvedChecker
from ..solved_data import SolvedDetail
from ..variables import LPVariables
from .initial_point_maker import ConstantInitialPointMaker, IInitialPointMaker
from .interior_point_method import InteriorPointMethod
from .search_direction_calculator import AbstractSearchDirectionCalculator
from .variable_updater import ArcVariableUpdater, LineVariableUpdater

logger = get_main_logger()


class InexactInteriorPointMethod(InteriorPointMethod, metaclass=ABCMeta):
    """Inexact に線形方程式を解きながら内点法を実施するクラス"""

    search_direction_calculator: AbstractSearchDirectionCalculator
    # 初期点時点で決定できるものや1回計算すればいいものは Attributes として使いまわす
    A_base_indexes: list[int] | None = None

    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
        solved_checker: SolvedChecker,
        initial_point_maker: IInitialPointMaker,
        search_direction_calculator: AbstractSearchDirectionCalculator,
    ):
        # TODO: solved_checker は Inexact 用でないといけない
        super().__init__(config_section, parameters, solved_checker, initial_point_maker)

        self.search_direction_calculator = search_direction_calculator
        logger.info(f"Linear system solver is {search_direction_calculator.linear_system_solver.__class__.__name__}.")
        logger.info(f"Search direction calculator is {search_direction_calculator.__class__.__name__}.")

    @property
    def beta(self) -> float:
        return self.parameters.INEXACT_COEF_OF_ARMIJO_RULE

    @property
    def sigma(self) -> float:
        return self.parameters.INEXACT_CENTERING_PARAMETER

    @property
    def eta(self) -> float:
        return self.parameters.INEXACT_TOLERANCE_OF_RESIDUAL_OF_LINEAR_SYSTEM

    @property
    def gamma_1(self) -> float:
        return self.parameters.INEXACT_COEF_NEIGHBORHOOD_DUALITY

    def make_initial_point(self, problem: LPS, v_0: LPVariables | None) -> LPVariables:
        if v_0 is not None:
            result = v_0
        else:
            result = self.initial_point_maker.make_initial_point(problem)

        # 初期点が近傍に入っていなければ, 新しく近傍に入る初期点を作成
        if not self.is_in_center_path_neighborhood(result, problem, self.calculate_gamma_2(result, problem)):
            logger.info("Initial points is not in neighborhood! Start with general initial point.")
            result = ConstantInitialPointMaker(self.parameters.INITIAL_POINT_SCALE).make_initial_point(problem)

        return result

    # def initial_problem_and_point(self, problem_0: LPS, v_0: LPVariables | None) -> tuple[LPS, LPVariables, list[int]]:
    #     """数値的に安定させるため, 与えられた問題と初期点に改良を加えて出力

    #     Args:
    #         problem_0 (LPS): 与えられた最初の問題
    #         v_0 (LPVariables): 初期点

    #     Returns:
    #         LPS: 修正した問題
    #         LPVariables: 修正した変数
    #         list[int]: 修正の過程で削除された制約の行の index
    #     """
    #     problem = problem_0
    #     v = v_0

    #     # Aの各行で正規化
    #     # 問題の最適解自体を変えてしまうので使用しないこととした
    #     # problem = problem.create_A_row_normalized()
    #     # logger.info("Problem is normalized for each A row.")

    #     # A の基底を取る関係で, 数値誤差で rank 落ちするような状況は避けたい.
    #     # なのでまず LU分解を施して数値誤差の範囲で0の行になるところは削除する
    #     # problem, remove_constraint_rows = problem.create_A_LU_factorized()
    #     # logger.info(f"Problem is LU factorized. number of removed rows: {len(remove_constraint_rows)}")
    #     # if remove_constraint_rows:
    #     #     logger.info(f"Removed constraint rows because of zero row: {remove_constraint_rows}")
    #     # v = v.remove_constraint_rows(remove_constraint_rows)
    #     remove_constraint_rows = []

    #     # 初期点が近傍に入っていなければ, 新しく近傍に入る初期点を作成
    #     if not self.is_in_center_path_neighborhood(v, problem, self.calculate_gamma_2(v, problem)):
    #         logger.info("Initial points is not in neighborhood! Start with general initial point.")
    #         v = ConstantInitialPointMaker(self.parameters.INITIAL_POINT_SCALE).make_initial_point(problem)

    #     return problem, v, remove_constraint_rows

    def calc_tolerance_for_inexact_first_derivative(self, v: LPVariables, problem: LPS) -> float:
        """一階微分を inexact に解く際の誤差許容度"""
        return self.eta * np.sqrt(v.mu / problem.n)

    def is_close_to_optimal(self, v: LPVariables, problem: LPS) -> bool:
        """最適解に近づいているか.
        最適解に近づくと閾値が低かったり condition number が大きくなったりで inexact solver が求解するのに時間がかかる.
        なので, 十分最適解に近づいたら exact に求解するようにするための判定

        Args:
            v (LPVariables): 現在の反復点
            problem (LPS): 現在の問題

        Returns:
            bool: inexact solver に求められる精度が 10^{-4} 未満であれば True
        """
        return v.mu < 1 / problem.n
        # return self.calc_tolerance_for_inexact_first_derivative(v, problem) < 10**(-4)

    def calc_first_derivatives(
        self,
        v: LPVariables,
        problem: LPS,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """一次微分の値を出力. MNES の定式化で計算するため中身を内点法のものから変更

        Returns:
            np.ndarray: x の一次微分
            np.ndarray: y の一次微分
            np.ndarray: s の一次微分
            float: 線形方程式を解いた際の誤差ノルム
        """
        right_hand_side = np.concatenate(
            [
                problem.residual_main_constraint(v.x),
                problem.residual_dual_constraint(v.y, v.s),
                v.x * v.s - self.sigma * v.mu,
            ]
        )

        # 求解
        tol = self.calc_tolerance_for_inexact_first_derivative(v, problem)
        x_dot, y_dot, s_dot, norm_residual = self.search_direction_calculator.run(v, problem, right_hand_side, tol)
        logger.debug(f"torelance: {tol}, ||M_1 z - sigma_1||: {norm_residual}")
        if norm_residual > tol:
            logger.warning(
                f"First derivative residual over tolerance! torelance: {tol}, ||M_1 z - sigma_1||: {norm_residual}"
            )

        return x_dot, y_dot, s_dot, norm_residual

    def calculate_gamma_2(self, v_0: LPVariables, problem: LPS) -> float:
        """近傍のパラメータで必要な gamma_2 の計算
        初期点での係数も含む

        Args:
            v_0 (LPVariables): 初期点
            problem (LPS): LP. 制約残差を取得するために使用

        Returns:
            float: 論文で定義された gamma_2
        """
        # mu_0 が 0 に近いと(Iterative Refinement でありうる) warning が出てうっとうしいので, 0 の時は場合分け
        if np.isclose(v_0.mu, 0):
            return np.inf

        resi_main = problem.residual_main_constraint(v_0.x)
        resi_dual = problem.residual_dual_constraint(v_0.y, v_0.s)
        norm_residuals = np.linalg.norm(np.concatenate([resi_main, resi_dual]))
        return self.parameters.INEXACT_COEF_NEIGHBORHOOD_CONSTRAINTS * norm_residuals / v_0.mu

    def is_in_center_path_neighborhood(self, v: LPVariables, problem: LPS, gamma_2: float) -> bool:
        """現在の点が論文上の近傍に入っているかを判定する関数

        Args:
            v (LPVariables): 現在の反復点. 初期点が近傍に入っているかを確認することもある.
            problem (LPS): LP. 制約残差を取得するために使用
            gamma_2 (float): 初期点の時点で計算されるパラメータ

        Returns:
            bool: 近傍の条件を満たしていれば True
        """
        mu = v.mu
        x_s_minus_gamma_mu = v.x * v.s - self.gamma_1 * mu
        if any(x_s_minus_gamma_mu < 0):
            return False

        # mu_0 が0に十分近いと warning が鬱陶しいので場合分け
        if gamma_2 == np.inf:
            return True

        resi_main = problem.residual_main_constraint(v.x)
        resi_dual = problem.residual_dual_constraint(v.y, v.s)
        return np.linalg.norm(np.concatenate([resi_main, resi_dual])) <= gamma_2 * mu

    def is_G_and_g_no_less_than_0(self, v_alpha: LPVariables, v: LPVariables, alpha: float) -> bool:
        """論文上の G^k, g^k が0以上になっているかを判定する関数
        理論的には gamma_2 が適切に設定されていれば中心パスに入るが,
        数値誤差でずれることもあるため正しく計算する

        Args:
            v_alpha (LPVariables): 与えられた step size だけ移動したと仮定した場合の変数
            v (LPVariables): 現在の反復点. 初期点が近傍に入っているかを確認することもある.
            alpha (float): step size

        Returns:
            bool: 近傍の条件を満たしていれば True
        """
        # G_i^k >= 0
        mu_alpha = v_alpha.mu
        x_s_minus_gamma_mu = v_alpha.x * v_alpha.s - self.gamma_1 * mu_alpha
        if any(x_s_minus_gamma_mu < 0):
            return False

        # g^k >= 0
        return mu_alpha >= (1 - alpha) * v.mu

    def is_h_no_less_than_0(self, v_alpha: LPVariables, v: LPVariables, alpha: float) -> bool:
        """論文上の h^k が0以上になっているかを判定する関数

        Args:
            v_alpha (LPVariables): 与えられた step size だけ移動したと仮定した場合の変数
            v (LPVariables): 現在の反復点
            alpha (float): step size

        Returns:
            bool: h^k(alpha) >= 0 なら True
        """
        return (1 - (1 - self.beta) * np.sin(alpha)) * v.mu - v_alpha.mu >= 0

    def log_constraints_residual_decreasing(self, v: LPVariables, problem: LPS, alpha: float, pre_r_b: np.ndarray):
        """制約残差が減っているかロギング

        Args:
            alpha (float): step size
        """
        r_b = problem.residual_main_constraint(v.x)
        r_c = problem.residual_dual_constraint(v.y, v.s)
        logger.debug(f"{indent}max r_b: {max(np.abs(r_b))}, r_c: {max(np.abs(r_c))}")


class InexactLineSearchIPM(InexactInteriorPointMethod):
    def _validate(self):
        """アルゴリズムが理論通り動くようにパラメータが設定されているか確認
        もし条件が満たされていなければ, アルゴリズムが収束する保障がないことを warning
        """
        is_sigma_more_than_eta = self.sigma > self.eta
        if not is_sigma_more_than_eta:
            logger.warning(f"sigma must be more than eta! sigma: {self.sigma}, eta: {self.eta}")

        is_beta_more_than_sigma_plus_eta = self.beta > self.sigma + self.eta
        if not is_beta_more_than_sigma_plus_eta:
            logger.warning(
                f"beta must be more than sigma + eta! beta: {self.beta}, sigma: {self.sigma}, eta: {self.eta}"
            )

        if not (is_sigma_more_than_eta and is_beta_more_than_sigma_plus_eta):
            logger.warning("Algorithm is not satisfied for convergence.")

    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
        solved_checker: SolvedChecker,
        initial_point_maker: IInitialPointMaker,
        search_direction_calculator: AbstractSearchDirectionCalculator,
    ):
        super().__init__(config_section, parameters, solved_checker, initial_point_maker, search_direction_calculator)
        self.variable_updater = LineVariableUpdater(self._delta_xs)
        self._validate()

    def is_iteration_number_reached_upper(self, iter_num: int, problem: LPS) -> bool:
        """反復回数が上限を超えたか

        本来は O(n^1.5 L) だが, 時間がかかって嫌なので設定どおりの値を使用

        Args:
            iter_num: 反復回数
        """
        upper = max(
            self.parameters.ITER_UPPER_COEF * (problem.n**1.5),
            self.parameters.ITER_UPPER,
        )
        # upper = self.parameters.ITER_UPPER
        return iter_num >= upper

    def run(self, problem_0: LPS, v_0: LPVariables | None) -> SolvedDetail:
        """反復で解く line-saarch の実行, 線形方程式は inexact に解く

        Args:
            problem_0 (LPS): 問題の最初の状態
            v_0 (LPVariables): 初期点

        Returns:
            SolvedDetail: 求解した結果
        """
        # 実行時間記録開始
        start_time = time.time()

        # 初期点の設定
        v_0 = self.make_initial_point(problem_0, v_0)
        self.log_initial_situation(problem_0, v_0)
        # 初期点時点で最適解だった場合, そのまま出力
        if self.solved_checker.run(v_0, problem_0):
            logger.info("Initial point satisfies solved condition.")
            aSolvedSummary = self.make_SolvedSummary(v_0, problem_0, True, 0, False, time.time() - start_time)
            return SolvedDetail(aSolvedSummary, v_0, problem_0, v_0, problem_0)

        # 初期点を現在の点として初期化
        v = v_0
        problem = problem_0

        mu = v.mu
        mu_0 = mu
        r_b = problem.residual_main_constraint(v.x)
        r_b_0 = r_b
        r_c = problem.residual_dual_constraint(v.y, v.s)
        r_c_0 = r_c
        gamma_2 = self.calculate_gamma_2(v, problem)

        iter_num = 0
        is_solved = self.solved_checker.run(v_0, problem_0)
        is_terminated = self.is_terminate(is_solved, iter_num, problem_0, time.time() - start_time)

        # SolvedDetail の出力
        lst_variables = [v]
        lst_mu = [mu_0]
        lst_alpha: list[float] = []
        lst_norm_vdot: list[float] = []
        lst_max_norm_main_const = [np.linalg.norm(r_b_0, ord=np.inf)]
        lst_max_norm_dual_const = [np.linalg.norm(r_c_0, ord=np.inf)]
        lst_residual_inexact_vdot: list[float] = []
        lst_tolerance_inexact_vdot: list[float] = []

        # while 文中で使用
        xs_ddot = np.zeros(problem.n)
        y_ddot = np.zeros(problem.m)

        # 収束条件を満たすまで反復する
        while not is_terminated:
            iter_num += 1
            logger.info(f"Iteration number: {iter_num}, mu: {mu}")
            logger.info(
                f"{indent}max_r_b: {np.linalg.norm(r_b, ord=np.inf)}, max_r_c: {np.linalg.norm(r_c, ord=np.inf)}"
            )

            # 探索方向の決定
            lst_tolerance_inexact_vdot.append(self.calc_tolerance_for_inexact_first_derivative(v, problem))

            x_dot, y_dot, s_dot, residual_first_derivative = self.calc_first_derivatives(v, problem)
            lst_norm_vdot.append(np.linalg.norm(np.concatenate([x_dot, y_dot, s_dot])))
            lst_residual_inexact_vdot.append(residual_first_derivative)

            # 近傍に入る step size になるまで Armijo のルールに従う
            alpha_x_max = self.variable_updater.max_step_size_guarantee_positive(v.x, x_dot, xs_ddot)
            alpha_s_max = self.variable_updater.max_step_size_guarantee_positive(v.s, s_dot, xs_ddot)
            alpha = min(alpha_x_max, alpha_s_max)
            logger.debug(f"{indent}Max step size: {alpha}")
            while alpha > self.min_step_size:
                v_alpha = LPVariables(
                    self.variable_updater.run(v.x, x_dot, xs_ddot, alpha),
                    self.variable_updater.run(v.y, y_dot, y_ddot, alpha),
                    self.variable_updater.run(v.s, s_dot, xs_ddot, alpha),
                )
                is_in_neighborhood = self.is_in_center_path_neighborhood(
                    v_alpha, problem, gamma_2
                ) and self.is_G_and_g_no_less_than_0(v_alpha, v, alpha)
                if is_in_neighborhood and self.is_h_no_less_than_0(v_alpha, v, alpha):
                    break
                alpha /= 2
            # while 文入らなかった場合に備えて新しく作成
            v = LPVariables(
                self.variable_updater.run(v.x, x_dot, xs_ddot, alpha),
                self.variable_updater.run(v.y, y_dot, y_ddot, alpha),
                self.variable_updater.run(v.s, s_dot, xs_ddot, alpha),
            )
            logger.info(f"{indent}Step size: {alpha}")

            # もし x, s が負になってしまった場合アルゴリズムが狂うので, 負になっていないか確認
            self.log_positive_variables_negativity(v)

            # 制約残渣更新
            pre_r_b = r_b
            pre_r_c = r_c
            r_b = problem.residual_main_constraint(v.x)
            r_c = problem.residual_dual_constraint(v.y, v.s)
            # 制約残差がどうなっているかlogging
            self.log_constraints_residual_decreasing(v, problem, alpha, pre_r_b)

            mu = v.mu

            # 記録用の値追加
            lst_variables.append(v)
            lst_mu.append(mu)
            lst_alpha.append(alpha)
            lst_max_norm_main_const.append(np.linalg.norm(r_b, ord=np.inf))
            lst_max_norm_dual_const.append(np.linalg.norm(r_c, ord=np.inf))

            # 停止条件更新
            is_solved = self.solved_checker.run(v, problem)
            is_terminated = self.is_terminate(
                is_solved,
                iter_num,
                problem,
                time.time() - start_time,
                alpha_x=alpha,
                alpha_s=alpha,
                pre_r_b=pre_r_b,
                pre_r_c=pre_r_c,
                r_b=r_b,
                r_c=r_c,
            )

        # 時間計測終了
        elapsed_time = time.time() - start_time

        # 出力の作成
        aSolvedSummary = self.make_SolvedSummary(
            v,
            problem,
            is_solved,
            iter_num,
            self.is_iteration_number_reached_upper(iter_num, problem),
            elapsed_time,
        )
        output = SolvedDetail(
            aSolvedSummary,
            v,
            problem,
            v_0,
            problem_0,
            lst_variables_by_iter=lst_variables,
            lst_main_step_size_by_iter=lst_alpha,
            lst_dual_step_size_by_iter=lst_alpha,
            lst_mu_by_iter=lst_mu,
            lst_norm_vdot_by_iter=lst_norm_vdot,
            lst_max_norm_main_constraint_by_iter=lst_max_norm_main_const,
            lst_max_norm_dual_constraint_by_iter=lst_max_norm_dual_const,
            lst_residual_inexact_vdot=lst_residual_inexact_vdot,
            lst_tolerance_inexact_vdot=lst_tolerance_inexact_vdot,
        )
        return output


class InexactArcSearchIPM(InexactInteriorPointMethod):
    def _validate(self):
        """アルゴリズムが理論通り動くようにパラメータが設定されているか確認
        もし条件が満たされていなければ, アルゴリズムが収束する保障がないことを warning
        """
        is_sigma_more_than_eta = self.sigma > self.eta
        if not is_sigma_more_than_eta:
            logger.warning(f"sigma must be more than eta! sigma: {self.sigma}, eta: {self.eta}")

        is_G_i_more_than_zero = (1 - self.gamma_1) * self.sigma > (1 + self.gamma_1) * self.eta
        if not is_G_i_more_than_zero:
            logger.warning(
                f"(1 - gamma_1) sigma must be more than (1 + gamma_1) eta! gamma_1: {self.gamma_1}, sigma: {self.sigma}, eta: {self.eta}"
            )

        is_beta_more_than_sigma_plus_eta = self.beta > self.sigma + self.eta
        if not is_beta_more_than_sigma_plus_eta:
            logger.warning(
                f"beta must be more than sigma + eta! beta: {self.beta}, sigma: {self.sigma}, eta: {self.eta}"
            )

        if not (is_sigma_more_than_eta and is_G_i_more_than_zero and is_beta_more_than_sigma_plus_eta):
            logger.warning("Algorithm is not satisfied for convergence.")

    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
        solved_checker: SolvedChecker,
        initial_point_maker: IInitialPointMaker,
        search_direction_calculator: AbstractSearchDirectionCalculator,
    ):
        super().__init__(config_section, parameters, solved_checker, initial_point_maker, search_direction_calculator)
        self.variable_updater = ArcVariableUpdater(self._delta_xs)
        self._validate()

    def is_iteration_number_reached_upper(self, iter_num: int, problem: LPS) -> bool:
        """反復回数が上限を超えたか

        本来は O(n^1.5 L) だが, 時間がかかって嫌なので設定どおりの値を使用

        Args:
            iter_num: 反復回数
        """
        upper = max(
            self.parameters.ITER_UPPER_COEF * (problem.n**1.5),
            self.parameters.ITER_UPPER,
        )
        # upper = self.parameters.ITER_UPPER
        return iter_num >= upper

    def calc_tolerance_for_inexact_second_derivative(self, v: LPVariables, problem: LPS) -> float:
        """二階微分を inexact に解く際の誤差許容度"""
        return self.calc_tolerance_for_inexact_first_derivative(v, problem)

    def calc_second_derivative(
        self,
        v: LPVariables,
        x_dot: np.ndarray,
        y_dot: np.ndarray,
        s_dot: np.ndarray,
        problem: LPS,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """二次微分の値を出力. OSS の定式化で計算するため中身を内点法のものから変更

        Returns:
            np.ndarray: x の二次微分
            np.ndarray: y の二次微分
            np.ndarray: s の二次微分
            float: 線形方程式を解いた際の誤差ノルム
        """
        rhs_elements_nonzero = -2 * x_dot * s_dot
        y_ddot_zero = np.zeros(problem.m)
        xs_ddot_zero = np.zeros(problem.n)

        # 二階微分の計算をしなくても threshold 以下に収まるなら, 計算時間短縮のために省く
        residual_with_0_vector = np.linalg.norm(rhs_elements_nonzero, ord=np.inf)
        tol_for_max_norm = self.eta * v.mu
        if residual_with_0_vector <= tol_for_max_norm:
            logger.info("RHS of second derivative is too small, so return zero vector.")
            logger.info(f"norm of RHS: {residual_with_0_vector}, tolerance: {tol_for_max_norm}")
            return xs_ddot_zero, y_ddot_zero, xs_ddot_zero, residual_with_0_vector

        # 求解
        right_hand_side = np.concatenate([np.zeros(problem.m + problem.n), rhs_elements_nonzero])
        tol = self.calc_tolerance_for_inexact_second_derivative(v, problem)
        x_ddot, y_ddot, s_ddot, norm_residual = self.search_direction_calculator.run(v, problem, right_hand_side, tol)
        logger.debug(f"tolerance: {tol}, ||M_2 z - sigma_2||: {norm_residual}")
        if norm_residual > tol:
            logger.warning(
                f"Second derivative residual over tolerance! tolerance: {tol}, ||M_2 z - sigma_2||: {norm_residual}"
            )

        # 誤差が0ベクトルの場合よりも大きい場合, 解を 0ベクトルに修正
        x_ddot_with_zero_y_ddot = rhs_elements_nonzero / v.s
        residual_with_zero_y_ddot = np.linalg.norm(problem.A @ x_ddot_with_zero_y_ddot)
        logger.debug(
            f"{indent}Norm of second derivative residual = {norm_residual}, ||RHS|| = {residual_with_zero_y_ddot}"
        )
        if norm_residual > residual_with_zero_y_ddot:
            logger.warning(
                f"{indent}Second derivative residual is too large. Second derivative is changed to zero vector."
            )
            return (
                x_ddot_with_zero_y_ddot,
                y_ddot_zero,
                xs_ddot_zero,
                residual_with_zero_y_ddot,
            )

        return x_ddot, y_ddot, s_ddot, norm_residual

    def run(self, problem_0: LPS, v_0: LPVariables | None) -> SolvedDetail:
        """反復で解く arc-saarch の実行, 線形方程式は inexact に解く

        Args:
            problem_0 (LPS): 問題の最初の状態
            v_0 (LPVariables): 初期点

        Returns:
            SolvedDetail: 求解した結果
        """
        # 実行時間記録開始
        start_time = time.time()

        # 初期点の設定
        v_0 = self.make_initial_point(problem_0, v_0)
        self.log_initial_situation(problem_0, v_0)
        # 初期点時点で最適解だった場合, そのまま出力
        if self.solved_checker.run(v_0, problem_0):
            logger.info("Initial point satisfies solved condition.")
            aSolvedSummary = self.make_SolvedSummary(v_0, problem_0, True, 0, False, time.time() - start_time)
            return SolvedDetail(aSolvedSummary, v_0, problem_0, v_0, problem_0)

        # 初期点を現在の点として初期化
        v = v_0
        problem = problem_0

        mu = v.mu
        mu_0 = mu
        r_b = problem.residual_main_constraint(v.x)
        r_b_0 = r_b
        r_c = problem.residual_dual_constraint(v.y, v.s)
        r_c_0 = r_c
        gamma_2 = self.calculate_gamma_2(v, problem)

        iter_num = 0
        is_solved = self.solved_checker.run(v_0, problem_0)
        is_terminated = self.is_terminate(is_solved, iter_num, problem_0, time.time() - start_time)

        # SolvedDetail の出力
        lst_variables = [v]
        lst_mu = [mu_0]
        lst_alpha: list[float] = []
        lst_norm_vdot: list[float] = []
        lst_norm_vddot: list[float] = []
        lst_max_norm_main_const = [np.linalg.norm(r_b_0, ord=np.inf)]
        lst_max_norm_dual_const = [np.linalg.norm(r_c_0, ord=np.inf)]
        lst_residual_inexact_vdot: list[float] = []
        lst_tolerance_inexact_vdot: list[float] = []
        lst_residual_inexact_vddot: list[float] = []
        lst_tolerance_inexact_vddot: list[float] = []

        # while 文中で使用
        # xs_ddot_zero = np.zeros(problem.n)
        # y_ddot_zero = np.zeros(problem.m)

        # 収束条件を満たすまで反復する
        while not is_terminated:
            iter_num += 1
            logger.info(f"Iteration number: {iter_num}, mu: {mu}")
            logger.info(
                f"{indent}max_r_b: {np.linalg.norm(r_b, ord=np.inf)}, max_r_c: {np.linalg.norm(r_c, ord=np.inf)}"
            )

            # 探索方向の決定
            lst_tolerance_inexact_vdot.append(self.calc_tolerance_for_inexact_first_derivative(v, problem))
            x_dot, y_dot, s_dot, residual_first_derivative = self.calc_first_derivatives(v, problem)
            lst_norm_vdot.append(np.linalg.norm(np.concatenate([x_dot, y_dot, s_dot])))
            lst_residual_inexact_vdot.append(residual_first_derivative)

            lst_tolerance_inexact_vddot.append(self.calc_tolerance_for_inexact_second_derivative(v, problem))

            x_ddot, y_ddot, s_ddot, residual_second_derivative = self.calc_second_derivative(
                v, x_dot, y_dot, s_dot, problem
            )
            lst_norm_vddot.append(np.linalg.norm(np.concatenate([x_ddot, y_ddot, s_ddot])))
            lst_residual_inexact_vddot.append(residual_second_derivative)

            # 近傍に入る step size になるまで Armijo のルールに従う
            alpha_x_max = self.variable_updater.max_step_size_guarantee_positive(v.x, x_dot, x_ddot)
            alpha_s_max = self.variable_updater.max_step_size_guarantee_positive(v.s, s_dot, s_ddot)
            alpha = min(alpha_x_max, alpha_s_max)
            logger.debug(f"{indent}Max step size: {alpha}")
            while alpha > self.min_step_size:
                v_alpha = LPVariables(
                    self.variable_updater.run(v.x, x_dot, x_ddot, alpha),
                    self.variable_updater.run(v.y, y_dot, y_ddot, alpha),
                    self.variable_updater.run(v.s, s_dot, s_ddot, alpha),
                )
                is_in_neighborhood = self.is_in_center_path_neighborhood(
                    v_alpha, problem, gamma_2
                ) and self.is_G_and_g_no_less_than_0(v_alpha, v, alpha)
                if is_in_neighborhood and self.is_h_no_less_than_0(v_alpha, v, alpha):
                    break
                alpha /= 2
            v = LPVariables(
                self.variable_updater.run(v.x, x_dot, x_ddot, alpha),
                self.variable_updater.run(v.y, y_dot, y_ddot, alpha),
                self.variable_updater.run(v.s, s_dot, s_ddot, alpha),
            )
            logger.info(f"{indent}Step size: {alpha}")

            # もし x, s が負になってしまった場合アルゴリズムが狂うので, 負になっていないか確認
            self.log_positive_variables_negativity(v)

            pre_r_b = r_b
            pre_r_c = r_c
            r_b = problem.residual_main_constraint(v.x)
            r_c = problem.residual_dual_constraint(v.y, v.s)
            # 制約残差がどうなっているかlogging
            self.log_constraints_residual_decreasing(v, problem, alpha, pre_r_b)

            mu = v.mu

            # 記録用の値追加
            lst_variables.append(v)
            lst_mu.append(mu)
            lst_alpha.append(alpha)
            lst_max_norm_main_const.append(np.linalg.norm(r_b, ord=np.inf))
            lst_max_norm_dual_const.append(np.linalg.norm(r_c, ord=np.inf))

            # 停止条件更新
            is_solved = self.solved_checker.run(v, problem)
            is_terminated = self.is_terminate(
                is_solved,
                iter_num,
                problem,
                time.time() - start_time,
                alpha_x=alpha,
                alpha_s=alpha,
                pre_r_b=pre_r_b,
                pre_r_c=pre_r_c,
                r_b=r_b,
                r_c=r_c,
            )

        # 時間計測終了
        elapsed_time = time.time() - start_time

        aSolvedSummary = self.make_SolvedSummary(
            v,
            problem,
            is_solved,
            iter_num,
            self.is_iteration_number_reached_upper(iter_num, problem),
            elapsed_time,
        )
        # problem_0 と比較する際に変数の次元が合わないと困るため復帰させる
        output = SolvedDetail(
            aSolvedSummary,
            v,
            problem,
            v_0,
            problem_0,
            lst_variables_by_iter=lst_variables,
            lst_main_step_size_by_iter=lst_alpha,
            lst_dual_step_size_by_iter=lst_alpha,
            lst_mu_by_iter=lst_mu,
            lst_norm_vdot_by_iter=lst_norm_vdot,
            lst_norm_vddot_by_iter=lst_norm_vddot,
            lst_max_norm_main_constraint_by_iter=lst_max_norm_main_const,
            lst_max_norm_dual_constraint_by_iter=lst_max_norm_dual_const,
            lst_residual_inexact_vdot=lst_residual_inexact_vdot,
            lst_tolerance_inexact_vdot=lst_tolerance_inexact_vdot,
            lst_residual_inexact_vddot=lst_residual_inexact_vddot,
            lst_tolerance_inexact_vddot=lst_tolerance_inexact_vddot,
        )
        return output
