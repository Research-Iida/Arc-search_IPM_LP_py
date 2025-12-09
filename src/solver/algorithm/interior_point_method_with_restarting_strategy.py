import time
from abc import ABCMeta, abstractmethod

import numpy as np

from ...logger import get_main_logger, indent
from ...problem import LinearProgrammingProblemStandard as LPS
from ..optimization_parameters import OptimizationParameters
from ..solved_checker import SolvedChecker
from ..solved_data import SolvedDetail
from ..variables import LPVariables
from .initial_point_maker import ConstantInitialPointMaker, IInitialPointMaker
from .interior_point_method import ExactInteriorPointMethod, MehrotraTypeIPM
from .variable_updater import ArcVariableUpdater

logger = get_main_logger()


class IPMWithRestartingStrategyBase(ExactInteriorPointMethod, metaclass=ABCMeta):
    @property
    def beta(self) -> float:
        return self.parameters.RESTART_COEF_RESTARTING

    @property
    def is_guaranteeing_main_residual_decreasing(self) -> bool:
        """主制約の減少を保証するように beta_k を決めるかを出力

        Proven クラスでは必ず True を返す必要があるのでパラメータ化
        """
        return self.parameters.RESTART_IS_GUARANTEEING_MAIN_RESIDUAL_DECREASING

    def beta_k(
        self,
        current_point: np.ndarray,
        previous_point: np.ndarray,
        problem: LPS,
        is_guaranteeing_main_residual_decreasing: bool,
    ) -> float:
        """restarting parameter に対象の値をかけ合わせたものを出力

        更新後の値が0に近すぎると探索方向において特異行列になってしまうため, 最低限正の値になるように修正を行う
        理論的には current_point が0より大きければ問題ないが, 計算上そのようになってしまう
        """
        delta = current_point - previous_point
        # TODO: 更新後の値が0に近づきすぎないような例外処理（発生は確認されていないが）
        result = self.beta / np.linalg.norm(delta / current_point, ord=np.inf)

        # 主問題の制約残差を減らすことを保証する場合にはさらに値を減少
        if is_guaranteeing_main_residual_decreasing:
            logger.debug(f"{indent}beta_k guarantees main residual decreasing")
            current_residual = problem.residual_main_constraint(current_point)
            diff_residual = current_residual - problem.residual_main_constraint(previous_point)
            idx_nonzero_diff = np.where(np.abs(diff_residual) > 10 ** (-6))
            if len(idx_nonzero_diff[0]):
                min_guaranteed_main_residual_decreasing_beta = min(
                    np.abs(current_residual[idx_nonzero_diff] / diff_residual[idx_nonzero_diff])
                )
                result = min(result, min_guaranteed_main_residual_decreasing_beta)

        return result

    @abstractmethod
    def restart_variable(self, v_current: LPVariables, problem: LPS, v_previous: LPVariables = None) -> LPVariables:
        """Nesterov restarting strategy を適用して, 変数の更新を行う

        Args:
            v_current (LPVariables): 更新前の変数組 (x, y, s)
            v_previous (LPVariables): v_current から1反復前の変数組.
                Noneの場合は, v_current をそのまま返す

        Returns:
            LPVariables: 更新後の変数組 (z, y, s)
        """
        if v_previous is None:
            return v_current

        beta_k = self.beta_k(v_current.x, v_previous.x, problem, self.is_guaranteeing_main_residual_decreasing)
        logger.info(f"{indent*2}beta_k: {beta_k}")
        if beta_k < 0:
            logger.warning("beta_k is negative!")
        z = v_current.x + beta_k * (v_current.x - v_previous.x)
        zero_indexes = np.where(z == 0)[0]
        if len(zero_indexes):
            logger.warning(f"{indent*2} zero indexes of z: {zero_indexes}")
        return LPVariables(z, v_current.y, v_current.s)


class ArcSearchIPMWithRestartingStrategy(IPMWithRestartingStrategyBase, MehrotraTypeIPM):
    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
        solved_checker: SolvedChecker,
        initial_point_maker: IInitialPointMaker,
    ):
        super().__init__(config_section, parameters, solved_checker, initial_point_maker)
        self.variable_updater = ArcVariableUpdater(self._delta_xs)

    def is_iteration_number_reached_upper(self, iter_num: int, problem: LPS) -> bool:
        """O(nL) のため, n倍かけた値

        Args:
            iter_num: 反復回数
        """
        return iter_num >= max(self.parameters.ITER_UPPER_COEF * problem.n, self.parameters.ITER_UPPER)

    def restart_variable(self, v_current: LPVariables, problem: LPS, v_previous: LPVariables = None) -> LPVariables:
        """Nesterov restarting strategy を適用して, 変数の更新を行う

        Args:
            v_current (LPVariables): 更新前の変数組 (x, y, s)
            v_previous (LPVariables): v_current から1反復前の変数組.
                Noneの場合は, v_current をそのまま返す

        Returns:
            LPVariables: 更新後の変数組 (z, y, s)
        """
        return super().restart_variable(v_current, problem, v_previous)

    def run(self, problem_0: LPS, v_0: LPVariables | None) -> SolvedDetail:
        """Nesterov の加速法を組み入れた内点法の実行

        変数の対応がわかりづらいのでメモ
            v: (x^k, y^k, s^k)
            v_pre: (x^k-1, lambda^k-1, s^k-1)
            v_restarted: (z^k, y^k, s^k)
            v_next: (x^k+1, y^k+1, s^k+1)
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
        # 1つ前の反復での点. 最初は何もなし
        v_pre = None

        mu = v.mu
        mu_0 = mu
        r_b = problem.residual_main_constraint(v.x)
        r_b_0 = r_b
        r_c = problem.residual_dual_constraint(v.y, v.s)
        r_c_0 = r_c

        iter_num = 0
        is_solved = self.solved_checker.run(v_0, problem_0)
        is_terminated = self.is_terminate(is_solved, iter_num, problem_0, time.time() - start_time)

        # SolvedDetail の出力
        lst_variables = [v]
        lst_mu = [mu_0]
        lst_alpha_z: list[float] = []
        lst_alpha_s: list[float] = []
        lst_norm_vdot: list[float] = []
        lst_max_norm_main_const = [np.linalg.norm(r_b_0, ord=np.inf)]
        lst_max_norm_dual_const = [np.linalg.norm(r_c_0, ord=np.inf)]

        # 収束条件を満たすまで反復する
        while not is_terminated:
            logger.info(f"Iteration number: {iter_num}, mu: {mu}")

            # 加速法による変数更新
            v_restarted = self.restart_variable(v, problem, v_pre)
            logger.debug(f"{indent}mu_z^k = {v_restarted.mu}")

            # 探索方向の決定
            z_dot, y_dot, s_dot = self.calc_first_derivatives(v_restarted, problem)
            lst_norm_vdot.append(np.linalg.norm(np.concatenate([z_dot, y_dot, s_dot])))

            z_ddot, y_ddot, s_ddot = self.calc_second_derivative(v_restarted, z_dot, y_dot, s_dot, problem)

            # step size の決定
            alpha_z_max = self.variable_updater.max_alpha_guarantee_positive(v_restarted.x, z_dot, z_ddot)
            alpha_s_max = self.variable_updater.max_alpha_guarantee_positive(v_restarted.s, s_dot, s_ddot)

            # 停止条件を満たしていれば終了
            v_max_step = LPVariables(
                self.variable_updater.run(v_restarted.x, z_dot, z_ddot, alpha_z_max),
                self.variable_updater.run(v_restarted.y, y_dot, y_ddot, alpha_s_max),
                self.variable_updater.run(v_restarted.s, s_dot, s_ddot, alpha_s_max),
            )
            is_solved = self.solved_checker.run(v_max_step, problem, mu_0=mu_0, r_b_0=r_b_0, r_c_0=r_c_0)
            if self.is_terminate(
                is_solved, iter_num, problem, time.time() - start_time, alpha_x=alpha_z_max, alpha_s=alpha_s_max
            ):
                # break しないが, while最後の is_terminate が True になるので問題ない
                logger.info(f"{indent}Variables satisfy terminate criteria with max step size.")
                alpha_z = alpha_z_max
                alpha_s = alpha_s_max
                v_next = v_max_step
            # 停止条件を満たしていなければ scale して次の反復へ
            else:
                alpha_z = self.scale_step_size(alpha_z_max, iter_num)
                alpha_s = self.scale_step_size(alpha_s_max, iter_num)
                v_next = LPVariables(
                    self.variable_updater.run(v_restarted.x, z_dot, z_ddot, alpha_z),
                    self.variable_updater.run(v_restarted.y, y_dot, y_ddot, alpha_s),
                    self.variable_updater.run(v_restarted.s, s_dot, s_ddot, alpha_s),
                )

            logger.info(f"{indent}Step size:")
            logger.info(f"{indent*2}alpha_z: {alpha_z:.2f}, alpha_s: {alpha_s:.2f}")

            # もし x, s が負になってしまった場合アルゴリズムが狂うので, 負になっていないか確認
            self.log_positive_variables_negativity(v_next)

            # 変数の更新
            v_pre = v
            v = v_next

            # 制約残差更新
            pre_r_b = r_b
            pre_r_c = r_c
            r_b = problem.residual_main_constraint(v.x)
            r_c = problem.residual_dual_constraint(v.y, v.s)
            # 制約残差がどうなっているかlogging
            rate_reduce = 1 - np.sin(alpha_z)
            logger.debug(f"{indent}1 - sin(alpha_z) = {rate_reduce}")
            for idx, (r_b_j, pre_r_b_j) in enumerate(zip(r_b, pre_r_b)):
                logger.debug(f"{indent}main residual {idx}: {pre_r_b_j} -> {r_b_j}")
                # 残渣が減っているか確認
                tolerance = 10 ** (-8)
                if not abs(r_b_j) < abs(pre_r_b_j) + tolerance:
                    logger.debug(f"{indent*2}norm of main residual is not decreasing with numerical error!")

                # 1 - sin(alpha_z) 以下に残差が減っているか確認
                is_theoretical = abs(abs(r_b_j) - abs(pre_r_b_j) * rate_reduce) <= tolerance
                logger.debug(f"{indent*2}||r_b(x^k+1)| - |r_b(x^k)| * (1 - sin(alpha))| <= 10^-8: {is_theoretical}")
                logger.debug(f"{indent*2}sign(r_b(x^k+1)) = sign(r_b(x^k)): {np.sign(r_b_j) == np.sign(pre_r_b_j)}")

            mu = v.mu
            # 反復回数追加
            iter_num += 1

            # 変数の値の記録
            lst_mu.append(mu)
            lst_alpha_z.append(alpha_z)
            lst_alpha_s.append(alpha_s)
            lst_variables.append(v)
            lst_max_norm_main_const.append(np.linalg.norm(r_b, ord=np.inf))
            lst_max_norm_dual_const.append(np.linalg.norm(r_c, ord=np.inf))

            is_solved = self.solved_checker.run(v, problem, mu_0=mu_0, r_b_0=r_b_0, r_c_0=r_c_0)
            is_terminated = self.is_terminate(
                is_solved,
                iter_num,
                problem,
                time.time() - start_time,
                alpha_x=alpha_z,
                alpha_s=alpha_s,
                pre_r_b=pre_r_b,
                pre_r_c=pre_r_c,
                r_b=r_b,
                r_c=r_c,
            )

        # 時間計測終了
        elapsed_time = time.time() - start_time

        # 出力の作成
        aSolvedSummary = self.make_SolvedSummary(
            v, problem, is_solved, iter_num, self.is_iteration_number_reached_upper(iter_num, problem), elapsed_time
        )
        output = SolvedDetail(
            aSolvedSummary,
            v,
            problem,
            v_0,
            problem_0,
            lst_variables_by_iter=lst_variables,
            # lst_merit_function_by_iter=lst_phi,
            lst_main_step_size_by_iter=lst_alpha_z,
            lst_dual_step_size_by_iter=lst_alpha_s,
            lst_mu_by_iter=lst_mu,
            lst_norm_vdot_by_iter=lst_norm_vdot,
            lst_max_norm_main_constraint_by_iter=lst_max_norm_main_const,
            lst_max_norm_dual_constraint_by_iter=lst_max_norm_dual_const,
        )
        return output


class ArcSearchIPMWithRestartingStrategyProven(IPMWithRestartingStrategyBase):
    """arc search restarting strategy の論文で収束することを証明したアルゴリズム"""

    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
        solved_checker: SolvedChecker,
        initial_point_maker: IInitialPointMaker,
    ):
        super().__init__(config_section, parameters, solved_checker, initial_point_maker)
        self.variable_updater = ArcVariableUpdater(self._delta_xs)

    @property
    def is_guaranteeing_main_residual_decreasing(self) -> bool:
        """主制約の減少を保証するように beta_k を決めるかを出力

        Proven クラスでは必ず True を返す
        """
        return True

    @property
    def theta(self) -> float:
        return self.parameters.RESTART_COEF_CENTER_PATH_NEIGHBORHOOD

    def is_in_center_path_neighborhood(self, v: LPVariables) -> bool:
        """中心パスの近傍に入っているか確認"""
        return np.linalg.norm(v.x * v.s - v.mu) <= self.theta * v.mu

    def is_iteration_number_reached_upper(self, iter_num: int, problem: LPS) -> bool:
        """O(nL) のため, n倍かけた値

        Args:
            iter_num: 反復回数
        """
        return iter_num >= max(self.parameters.ITER_UPPER_COEF * problem.n, self.parameters.ITER_UPPER)

    def make_initial_point(self, problem: LPS, v_0: LPVariables | None) -> LPVariables:
        if v_0 is None:
            result = v_0
        else:
            result = self.initial_point_maker.make_initial_point(problem)

        # 初期点が近傍に入っていなければ, 理論的収束性を担保できない
        if not self.is_in_center_path_neighborhood(result):
            logger.info("Initial point is not in neighborhood! Start with general initial point.")
            result = ConstantInitialPointMaker(self.parameters.INITIAL_POINT_SCALE).make_initial_point(problem)

        return result

    def run(self, problem_0: LPS, v_0: LPVariables | None) -> SolvedDetail:
        """Nesterov の加速法を組み入れた内点法の実行
        理論的な証明を施したもの

        変数の対応がわかりづらいのでメモ
            v: (x^k, y^k, s^k)
            v_pre: (x^k-1, y^k-1, s^k-1)
            v_restarted: (z^k, y^k, s^k)
            v_next: (x^k+1, y^k+1, s^k+1)
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
        # 1つ前の反復での点. 最初は何もなし
        v_pre = None

        mu = v.mu
        mu_0 = mu
        r_b = problem.residual_main_constraint(v.x)
        r_b_0 = r_b
        r_c = problem.residual_dual_constraint(v.y, v.s)
        r_c_0 = r_c

        iter_num = 0
        is_solved = self.solved_checker.run(v_0, problem_0)
        is_terminated = self.is_terminate(is_solved, iter_num, problem_0, time.time() - start_time)

        # SolvedDetail の出力
        lst_variables = [v]
        # lst_phi = []
        lst_mu = [mu_0]
        lst_alpha = []
        lst_norm_vdot = []
        lst_max_norm_main_const = [np.linalg.norm(r_b_0, ord=np.inf)]
        lst_max_norm_dual_const = [np.linalg.norm(r_c_0, ord=np.inf)]

        # 収束条件を満たすまで反復する
        while not is_terminated:
            iter_num += 1
            logger.info(f"Iteration number: {iter_num}, mu: {mu}")
            logger.info(f"{indent}In neighborhood: {self.is_in_center_path_neighborhood(v)}")

            # 加速法による変数更新
            v_restarted = self.restart_variable(v, problem, v_pre)
            mu_z = v_restarted.mu
            logger.info(f"{indent}mu_z^k = {mu_z}")

            # 進行方向 v_dot のノルム記録
            v_dot = self.calc_first_derivatives(v_restarted, problem)
            lst_norm_vdot.append(np.linalg.norm(np.concatenate(v_dot)))

            # 反復点更新
            z = v_restarted.x
            # 探索方向の決定
            z_dot, y_dot, s_dot = self.calc_first_derivatives(v_restarted, problem)
            # 二階微分における計算は sigma などは不要
            z_ddot, y_ddot, s_ddot = self.calc_second_derivative(v_restarted, z_dot, y_dot, s_dot, problem, sigma=0)

            # step size の決定, scaling は近傍に入るようにすれば正の値になるので不要
            alpha_z = self.variable_updater.max_alpha_guarantee_positive(z, z_dot, z_ddot)
            alpha_s = self.variable_updater.max_alpha_guarantee_positive(v_restarted.s, s_dot, s_ddot)
            alpha = min(alpha_z, alpha_s)

            # 近傍に入る step size になるまでアルミホのルールに従う
            while alpha > self.min_step_size:
                v_alpha = LPVariables(
                    self.variable_updater.run(z, z_dot, z_ddot, alpha),
                    self.variable_updater.run(v_restarted.y, y_dot, y_ddot, alpha),
                    self.variable_updater.run(v_restarted.s, s_dot, s_ddot, alpha),
                )
                if self.is_in_center_path_neighborhood_on_next_point(v_alpha, alpha, mu_z):
                    break
                alpha /= 2
            logger.info(f"{indent}Step size:")
            logger.info(f"{indent*2} alpha: {alpha:.2f}")

            # mu の減少性を担保するために修正
            v_next = self.next_iteration_point_with_mu_decreasing(v_alpha, problem, alpha, mu)

            # もし x, s が負になってしまった場合アルゴリズムが狂うので, 負になっていないか確認
            self.log_positive_variables_negativity(v_next)
            logger.debug(f"mu_k+1 = {v_next.mu}, mu_k * (1 - sin(alpha)) = {mu * (1 - np.sin(alpha))}")

            # 変数の更新
            v_pre = v
            v = v_next

            # 制約残差更新
            pre_r_b = r_b
            pre_r_c = r_c
            r_b = problem.residual_main_constraint(v.x)
            r_c = problem.residual_dual_constraint(v.y, v.s)
            for i, (r_b_i, pre_r_b_i) in enumerate(zip(r_b, pre_r_b)):
                logger.debug(f"{indent}main constraint residual {i}: {pre_r_b_i} -> {r_b_i}")
                if pre_r_b_i != 0:
                    logger.debug(f"{indent}r_b(x^k)_{i} / r_b(x^k-1)_{i} : {r_b_i / pre_r_b_i}")
                    logger.debug(
                        f"{indent*2}sign(r_b(x^k+1)) = sign(r_b(x^k)): {np.sign(r_b_i) == np.sign(pre_r_b_i)}"
                    )

            mu = v.mu

            # 変数の値の記録
            lst_mu.append(mu)
            lst_alpha.append(alpha)
            lst_variables.append(v)
            lst_max_norm_main_const.append(np.linalg.norm(r_b, ord=np.inf))
            lst_max_norm_dual_const.append(np.linalg.norm(r_c, ord=np.inf))

            # 停止条件更新
            is_solved = self.solved_checker.run(v, problem, mu_0=mu_0, r_b_0=r_b_0, r_c_0=r_c_0)
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

        logger.debug(f"x: {v.x}")
        logger.debug(f"lambda: {v.y}")
        logger.debug(f"s: {v.s}")

        # 出力の作成
        aSolvedSummary = self.make_SolvedSummary(
            v, problem, is_solved, iter_num, self.is_iteration_number_reached_upper(iter_num, problem), elapsed_time
        )
        output = SolvedDetail(
            aSolvedSummary,
            v,
            problem,
            v_0,
            problem_0,
            lst_variables_by_iter=lst_variables,
            # lst_merit_function_by_iter=lst_phi,
            lst_main_step_size_by_iter=lst_alpha,
            lst_dual_step_size_by_iter=lst_alpha,
            lst_mu_by_iter=lst_mu,
            lst_norm_vdot_by_iter=lst_norm_vdot,
            lst_max_norm_main_constraint_by_iter=lst_max_norm_main_const,
            lst_max_norm_dual_constraint_by_iter=lst_max_norm_dual_const,
        )
        return output

    def restart_variable(self, v_current: LPVariables, problem: LPS, v_previous: LPVariables = None) -> LPVariables:
        """Nesterov restarting strategy を適用して, 変数の更新を行う

        Args:
            v_current (LPVariables): 更新前の変数組 (x, y, s)
            v_previous (LPVariables): v_current から1反復前の変数組.
                Noneの場合は, v_current をそのまま返す

        Returns:
            LPVariables: 更新後の変数組 (z, y, s)
        """
        v_restarted = super().restart_variable(v_current, problem, v_previous)

        # 次の反復点が近傍に入っていれば, 変数の更新を行う
        if self.is_in_center_path_neighborhood(v_restarted):
            return v_restarted
        logger.info("Restarted point is not in neighborhood. Current point is used.")
        return v_current

    def is_in_center_path_neighborhood_on_next_point(self, v_alpha: LPVariables, alpha: float, mu_z: float) -> bool:
        """alpha によって更新された値が近傍に入っているかを確認"""
        base_mu = (1 - np.sin(alpha)) * mu_z
        return np.linalg.norm(v_alpha.x * v_alpha.s - base_mu) <= 2 * self.theta * base_mu

    def next_iteration_point_with_mu_decreasing(
        self, v_alpha: LPVariables, problem: LPS, alpha: float, mu: float
    ) -> LPVariables:
        """step size alpha で移動させた後に mu が減少することを保証するために移動させた点"""
        right_hand_side = np.concatenate(
            [np.zeros(problem.m + problem.n), ((1 - np.sin(alpha)) * mu - v_alpha.x * v_alpha.s)]
        )
        delta_x, delta_lambda, delta_s, _ = self.search_direction_calculator.run(v_alpha, problem, right_hand_side)
        return LPVariables(v_alpha.x + delta_x, v_alpha.y + delta_lambda, v_alpha.s + delta_s)
