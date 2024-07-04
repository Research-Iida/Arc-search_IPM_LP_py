import abc
import time

import numpy as np

from ...logger import get_main_logger, indent
from ...problem import LinearProgrammingProblemStandard as LPS
from ..linear_system_solver.exact_linear_system_solver import ExactLinearSystemSolver
from ..optimization_parameters import OptimizationParameters
from ..solved_checker import SolvedChecker
from ..solved_data import SolvedDetail
from ..variables import LPVariables
from .algorithm import ILPSolvingAlgoritm
from .initial_point_maker import IInitialPointMaker
from .search_direction_calculator import AbstractSearchDirectionCalculator, NESSearchDirectionCalculator
from .variable_updater import ArcVariableUpdater, LineVariableUpdater, VariableUpdater

logger = get_main_logger()


class InteriorPointMethod(ILPSolvingAlgoritm, metaclass=abc.ABCMeta):
    """LPを内点法で解く際のインターフェース"""

    variable_updater: VariableUpdater

    @property
    def _delta_xs(self) -> float:
        return self.parameters.IPM_COEF_GUARANTEEING_XS_POSITIVENESS

    @property
    def _epsilon_x(self) -> float:
        return self.parameters.IPM_LOWER_BOUND_OF_X_TRUNCATION

    @abc.abstractmethod
    def make_initial_point(self, problem: LPS, v_0: LPVariables | None) -> LPVariables:
        """初期点の決定. アルゴリズムによっては複雑な設定が必要になるので, 抽象メソッド

        Args:
            problem (LPS): 初期点作成対象のLP
            v_0 (LPVariables): 与えられた場合の初期点. ある場合とない場合で処理が異なる

        Returns:
            LPVariables: 初期点
        """
        pass

    @abc.abstractmethod
    def is_iteration_number_reached_upper(self, iter_num: int, problem: LPS) -> bool:
        """反復回数が上限に達したか. 上限以上の値であれば True の想定

        アルゴリズムによって反復回数上限が異なるため, abstract method

        Args:
            iter_num: 反復回数
        """
        pass

    def is_not_decrease_residuals(
        self, pre_r_b: np.ndarray, pre_r_c: np.ndarray, post_r_b: np.ndarray, post_r_c: np.ndarray
    ) -> bool:
        """制約の残渣においていずれかが減少せず, 加えて前の反復点より10倍以上小さくなった制約が存在しないか確認

        もし上記の条件のように小さくなっていなければ, 反復を停止させる
        参照："Arc-Search Techniques for Interior-Point Methods" Section 7.3.10
        """
        # 制約の残渣がいずれかが減少していれば停止しない
        if np.any(pre_r_b > post_r_b) or np.any(pre_r_c > post_r_c):
            return False

        # 制約残渣のいずれかが前の残渣より大きくなりすぎてなければ停止しない
        is_ten_times_decrease_r_b = np.any(10 * pre_r_b <= post_r_b)
        is_ten_times_decrease_r_c = np.any(10 * pre_r_c <= post_r_c)
        if is_ten_times_decrease_r_b or is_ten_times_decrease_r_c:
            return False
        # 上記の条件を満たさなければ停止
        return True

    def is_terminate(
        self,
        is_solved: bool,
        iter_num: int,
        problem: LPS,
        elapsed_time: float,
        alpha_x: float = None,
        alpha_s: float = None,
        pre_r_b: np.ndarray = None,
        pre_r_c: np.ndarray = None,
        r_b: np.ndarray = None,
        r_c: np.ndarray = None,
    ) -> bool:
        """アルゴリズムが停止条件を満たしたか否か

        停止条件は以下のいずれか
            * 最適性を満たす
            * 反復回数が上限に達した
            * 計算時間が上限に達した
            * step size が小さすぎる
            * 制約残渣のどちらかが減少せず, 加えてどちらかが前の反復点の10倍より小さくなっていない

        Args:
            alpha_x: x の step size. 反復0回目では進んでいないため, 判定の対象にならない
            alpha_s: s の step size. 反復0回目では進んでいないため, 判定の対象にならない
            pre_r_b: 前回の反復での主問題の制約違反度. 反復0回目では判定の対象にならない
            pre_r_c: 前回の反復での双対問題の制約違反度. 反復0回目では判定の対象にならない
        """
        if is_solved:
            logger.info(f"{indent}Variables satisfy the solution condition.")
            return True

        if self.is_iteration_number_reached_upper(iter_num, problem):
            logger.warning(f"{indent}Algorithm terminates iteration upper bound.")
            return True

        if self.is_calculation_time_reached_upper(elapsed_time):
            logger.warning(f"{indent}Algorithm terminates calculation time upper bound.")
            return True

        if alpha_x is not None and alpha_s is not None:
            if alpha_x < self.min_step_size and alpha_s < self.min_step_size:
                logger.warning(f"{indent}Step size is too small.")
                return True

        # 1つ前の制約の情報が入っているならば判定する
        # TODO: 制約残差の減少を保証しないアルゴリズムもある(Restartとか)ので, この停止条件は含めるべきか要検討
        if pre_r_b is not None and pre_r_c is not None and r_b is not None and r_c is not None:
            if self.is_not_decrease_residuals(pre_r_b, pre_r_c, r_b, r_c):
                logger.warning(f"{indent}Constraint residuals are not decreasing.")
                return True

        return False


class ExactInteriorPointMethod(InteriorPointMethod, metaclass=abc.ABCMeta):
    """線形方程式を正確に解くことを前提とした内点法"""

    search_direction_calculator: AbstractSearchDirectionCalculator

    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
        solved_checker: SolvedChecker,
        initial_point_maker: IInitialPointMaker,
    ):
        """インスタンス初期化

        Args:
            config_section (str): 設定ファイルのセクション名.
                logging にも使用するので文字列で取得しておく
        """
        super().__init__(config_section, parameters, solved_checker, initial_point_maker)

        self.search_direction_calculator = NESSearchDirectionCalculator(ExactLinearSystemSolver())

    def make_initial_point(self, problem: LPS, v_0: LPVariables | None) -> LPVariables:
        if v_0 is not None:
            result = v_0
        else:
            result = self.initial_point_maker.make_initial_point(problem)

        return result

    def calc_first_derivatives(
        self,
        v: LPVariables,
        problem: LPS,
        coef_matrix: np.ndarray | None = None,
        right_hand_side: np.ndarray | None = None,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """一次微分の値を出力

        Args:
            coef_matrix: 係数行列. すでに求められていれば使用する
            right_hand_side: 右辺のベクトル. すでに求められていれば使用する

        Returns:
            np.ndarray: x の一次微分
            np.ndarray: y の一次微分
            np.ndarray: s の一次微分
        """
        right_hand_side = np.concatenate(
            [problem.residual_main_constraint(v.x), problem.residual_dual_constraint(v.y, v.s), v.x * v.s]
        )

        # 求解
        x_dot, y_dot, s_dot, _ = self.search_direction_calculator.run(v, problem, right_hand_side)
        return x_dot, y_dot, s_dot

    def centering_parameter(self, v: LPVariables, x_dot: np.ndarray, s_dot: np.ndarray, mu: float = None) -> float:
        """centering parameter sigma の出力

        Args:
            mu: duality measure. x,s からも出力可能
        """
        if mu is None:
            mu = v.mu
        dim = v.x.shape[0]

        # mu^a の取得
        alpha_x = self.variable_updater.max_alpha_guarantee_positive_with_line(v.x, x_dot)
        x_minus_alpha_x_dot = v.x - alpha_x * x_dot
        alpha_s = self.variable_updater.max_alpha_guarantee_positive_with_line(v.s, s_dot)
        s_minus_alpha_s_dot = v.s - alpha_s * s_dot
        mu_a = x_minus_alpha_x_dot.T @ s_minus_alpha_s_dot / dim

        return (mu_a / mu) ** 3

    def calc_second_derivative(
        self,
        v: LPVariables,
        x_dot: np.ndarray,
        y_dot: np.ndarray,
        s_dot: np.ndarray,
        problem: LPS,
        *args,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """二次微分の値を出力

        Args:
            coef_matrix: 係数行列. すでに求められていれば使用する
            right_hand_side: 右辺のベクトル. すでに求められていれば使用する

        Returns:
            np.ndarray: x の二次微分
            np.ndarray: y の二次微分
            np.ndarray: s の二次微分
        """
        # Mehrotra-type のように perturbation する場合とそうでない場合があるので, sigma は入力として受け付ける
        if "sigma" in kwargs:
            sigma = kwargs["sigma"]
        else:
            sigma = self.centering_parameter(v, x_dot, s_dot, v.mu)

        right_hand_side = np.concatenate([np.zeros(problem.m + problem.n), sigma * v.mu - 2 * x_dot * s_dot])
        x_ddot, y_ddot, s_ddot, _ = self.search_direction_calculator.run(v, problem, right_hand_side)

        return x_ddot, y_ddot, s_ddot

    def indexes_x_not_zero_neighborhood(self, x: np.ndarray) -> list[int]:
        """x が0に近くなりすぎると step size を大きくとれなくなるため,
        0に近くなりすぎたxを削除するために, xが0に近くない添え字を取得する

        現在は0に近くなったxを削除していないのでxの添え字すべてを出力
        """
        # return np.where(x > self._epsilon_x)[0].tolist()
        return range(len(x))


class MehrotraTypeIPM(ExactInteriorPointMethod, metaclass=abc.ABCMeta):
    """Mehrotra と同じ種類のIPM
    アルゴリズムが異なってくるので別にした
    """

    def run(self, problem_0: LPS, v_0: LPVariables | None) -> SolvedDetail:
        """反復で解く内点法の実行
        アルゴリズムは基本的に異なるため abstractmethod.
        ただし ArcSearchIPM と LineSearchIPM は同じアルゴリズムとなるため,
        共通のものをここにおく
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

        iter_num = 0
        is_solved = self.solved_checker.run(v_0, problem_0)
        is_terminated = self.is_terminate(is_solved, iter_num, problem_0, time.time() - start_time)

        # SolvedDetail の出力
        lst_variables = [v]
        lst_mu = [mu_0]
        lst_alpha_x: list[float] = []
        lst_alpha_s: list[float] = []
        lst_norm_vdot: list[np.ndarray] = []
        lst_norm_vddot: list[np.ndarray] = []
        lst_max_norm_main_const = [np.linalg.norm(r_b_0, ord=np.inf)]
        lst_max_norm_dual_const = [np.linalg.norm(r_c_0, ord=np.inf)]

        # 収束条件を満たすまで反復する
        while not is_terminated:
            iter_num += 1
            logger.info(f"Iteration number: {iter_num}, mu: {mu}")

            # 探索方向の決定
            x_dot, y_dot, s_dot = self.calc_first_derivatives(v, problem)
            lst_norm_vdot.append(np.linalg.norm(np.concatenate([x_dot, y_dot, s_dot])))

            x_ddot, y_ddot, s_ddot = self.calc_second_derivative(v, x_dot, y_dot, s_dot, problem)
            lst_norm_vdot.append(np.linalg.norm(np.concatenate([x_ddot, y_ddot, s_ddot])))

            # step size の決定
            alpha_x_max = self.variable_updater.max_step_size_guarantee_positive(v.x, x_dot, x_ddot)
            alpha_s_max = self.variable_updater.max_step_size_guarantee_positive(v.s, s_dot, s_ddot)

            # 停止条件を満たしていれば終了
            v_max_step = LPVariables(
                self.variable_updater.run(v.x, x_dot, x_ddot, alpha_x_max),
                self.variable_updater.run(v.y, y_dot, y_ddot, alpha_s_max),
                self.variable_updater.run(v.s, s_dot, s_ddot, alpha_s_max),
            )
            is_solved = self.solved_checker.run(v_max_step, problem, mu_0=mu_0, r_b_0=r_b_0, r_c_0=r_c_0)
            if self.is_terminate(
                is_solved, iter_num, problem, time.time() - start_time, alpha_x=alpha_x_max, alpha_s=alpha_s_max
            ):
                # break しないが, while最後の is_terminate が True になるので問題ない
                logger.info(f"{indent}Variables satisfy terminate criteria with max step size.")
                alpha_x = alpha_x_max
                alpha_s = alpha_s_max
                v = v_max_step
            # 停止条件を満たしていなければ scale して次の反復へ
            else:
                alpha_x = self.scale_step_size(alpha_x_max, iter_num)
                alpha_s = self.scale_step_size(alpha_s_max, iter_num)
                v = LPVariables(
                    self.variable_updater.run(v.x, x_dot, x_ddot, alpha_x),
                    self.variable_updater.run(v.y, y_dot, y_ddot, alpha_s),
                    self.variable_updater.run(v.s, s_dot, s_ddot, alpha_s),
                )

            logger.info(f"{indent}Step size:")
            logger.info(f"{indent*2}alpha_x: {alpha_x:.2f}, alpha_s: {alpha_s:.2f}")

            # もし x, s が負になってしまった場合アルゴリズムが狂うので, 負になっていないか確認
            self.log_positive_variables_negativity(v)

            # xが0近くになったら変数を削除
            leave_cols = self.indexes_x_not_zero_neighborhood(v.x)
            x = v.x[leave_cols]
            A = problem.A[:, leave_cols]
            c = problem.c[leave_cols]
            s = v.s[leave_cols]
            v = LPVariables(x, v.y, s)
            problem = LPS(A, problem.b, c, problem.name)
            logger.debug(f"{indent}Dimension: n: {problem.n}, m: {problem.m}")
            # 制約残渣更新
            pre_r_b = r_b
            pre_r_c = r_c[leave_cols]
            r_b = problem.residual_main_constraint(x)
            r_c = problem.residual_dual_constraint(v.y, s)
            # 制約残差がどうなっているかlogging
            rate_reduce = 1 - np.sin(alpha_x)
            logger.debug(f"{indent}1 - sin(alpha_x) = {rate_reduce}")
            mu = v.mu

            # 記録用の値追加
            lst_variables.append(v)
            lst_mu.append(mu)
            lst_alpha_x.append(alpha_x)
            lst_alpha_s.append(alpha_s)
            lst_max_norm_main_const.append(np.linalg.norm(r_b, ord=np.inf))
            lst_max_norm_dual_const.append(np.linalg.norm(r_c, ord=np.inf))

            is_solved = self.solved_checker.run(v, problem, mu_0=mu_0, r_b_0=r_b_0, r_c_0=r_c_0)
            is_terminated = self.is_terminate(
                is_solved,
                iter_num,
                problem,
                time.time() - start_time,
                alpha_x,
                alpha_s,
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
            lst_main_step_size_by_iter=lst_alpha_x,
            lst_dual_step_size_by_iter=lst_alpha_s,
            lst_mu_by_iter=lst_mu,
            lst_norm_vdot_by_iter=lst_norm_vdot,
            lst_norm_vddot_by_iter=lst_norm_vddot,
            lst_max_norm_main_constraint_by_iter=lst_max_norm_main_const,
            lst_max_norm_dual_constraint_by_iter=lst_max_norm_dual_const,
        )
        return output


class LineSearchIPM(MehrotraTypeIPM):
    """LPを解く line search アルゴリズムの実装"""

    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
        solved_checker: SolvedChecker,
        initial_point_maker: IInitialPointMaker,
    ):
        super().__init__(config_section, parameters, solved_checker, initial_point_maker)

        self.variable_updater = LineVariableUpdater(self._delta_xs)

    def is_iteration_number_reached_upper(self, iter_num: int, problem: LPS) -> bool:
        """反復回数上限対応

        O(nL) だが, 繰り返し過ぎても時間かかるだけなので上限をパラメータで決める

        Args:
            iter_num: 反復回数
        """
        # iter_upper = max(self.parameters.ITER_UPPER_COEF * problem.n, self.parameters.ITER_UPPER)
        iter_upper = self.parameters.ITER_UPPER
        return iter_num >= iter_upper


class ArcSearchIPM(MehrotraTypeIPM):
    """LPを解く arc-search アルゴリズムの実装"""

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
        """反復回数上限対応

        O(nL) だが, 繰り返し過ぎても時間かかるだけなので上限をパラメータで決める

        Args:
            iter_num: 反復回数
        """
        # iter_upper = max(self.parameters.ITER_UPPER_COEF * problem.n, self.parameters.ITER_UPPER)
        iter_upper = self.parameters.ITER_UPPER
        return iter_num >= iter_upper
