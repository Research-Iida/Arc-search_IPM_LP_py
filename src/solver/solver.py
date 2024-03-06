"""solver に関する module

今後ソルバーの設定を変えることで実験することが頻繁に起こるため,
インターフェイスを使用して変更に対して柔軟な設計をできるようにしておく
"""
from abc import ABCMeta, abstractmethod

import numpy as np

from ..logger import get_main_logger, indent
from ..problem import LinearProgrammingProblemStandard as LPS
from .variables import LPVariables
from .solved_checker import SolvedChecker, RelativeSolvedChecker, AbsoluteSolvedChecker
from .optimization_parameters import OptimizationParameters
from .solved_data import SolvedSummary, SolvedDetail

logger = get_main_logger()


class LPSolver(metaclass=ABCMeta):
    """LPを解くためのソルバーに関する抽象クラス
    """
    solved_checker: SolvedChecker

    def _set_config_and_parameters(self, config_section: str):
        self.config_section = config_section
        self.parameters = OptimizationParameters.import_(config_section)

    @property
    def is_stopping_criteria_relative(self) -> bool:
        """停止条件を relative なもの（数値実験上は効率がよいとされている）に設定するか
        """
        return self.parameters.IS_STOPPING_CRITERIA_RELATIVE

    def __init__(
        self,
        config_section: str,
        solved_checker: SolvedChecker | None,
    ):
        """インスタンス初期化

        Args:
            config_section (str): 設定ファイルのセクション名.
                logging にも使用するので文字列で取得しておく
        """
        self._set_config_and_parameters(config_section)

        if solved_checker is None:
            threshold = self.parameters.STOP_CRITERIA_PARAMETER
            if self.is_stopping_criteria_relative:
                self.solved_checker = RelativeSolvedChecker(threshold, self.parameters.THRESHOLD_XS_NEGATIVE)
            else:
                self.solved_checker = AbsoluteSolvedChecker(threshold, self.parameters.THRESHOLD_XS_NEGATIVE)
        else:
            self.solved_checker = solved_checker

    @property
    def min_step_size(self) -> float:
        return self.parameters.MIN_STEP_SIZE

    @property
    def initial_point_scale(self) -> int:
        return self.parameters.INITIAL_POINT_SCALE

    def run(self, problem: LPS, v_0: LPVariables = None) -> SolvedDetail:
        """入力されたLPに対してアルゴリズムを実行

        Aの行数とbの次元数, およびAの列数とcの次元数が異なる場合, エラーを起こす

        Args:
            problem: LPS における係数群をまとめたクラスインスタンス
            v_0: 初期点

        Returns:
            SolvedDetail: 最適解に関する情報をまとめたインスタンス
        """
        solver_name = self.__class__.__name__
        logger.info(f"[{solver_name}] [{self.config_section}] Start solving {problem.name}.")

        logger.info("Logging problem information.")
        self.log_initial_problem_information(problem)

        # 初期点が設定されていなければ問題から設定
        if v_0 is None:
            v_0 = self.initial_variables(problem)

        # 初期点の設定
        logger.info("Logging initial situation.")
        self.log_initial_situation(v_0, problem)
        # 初期点時点で最適解だった場合, そのまま出力
        if self.solved_checker.run(v_0, problem):
            logger.info("Initial point satisfies solved condition.")
            aSolvedSummary = self.make_SolvedSummary(v_0, problem, True, 0, False, 0)
            return SolvedDetail(aSolvedSummary, v_0, problem, v_0, problem)

        # アルゴリズムの実行
        try:
            aSolvedDetail = self.run_algorithm(problem, v_0)

            logger.info(f"[{solver_name}] [{self.config_section}] End solving {problem.name}.")
            self.log_solved_data(aSolvedDetail)
        # 計算上でエラーが起きても計算が止まらないようにエラー文を生成だけして結果を書き込む
        except Exception as e:
            logger.exception("Error occured - ", exc_info=e)
            aSolvedSummary = SolvedSummary(
                problem.name,
                solver_name,
                self.config_section,
                True,
                problem.n,
                problem.m,
                False
            )
            aSolvedDetail = SolvedDetail(aSolvedSummary, v_0, problem, v_0, problem)

        # 求解不可能だった場合, ログに残す
        if not aSolvedDetail.aSolvedSummary.is_solved:
            logger.warning(f"{solver_name} cannot solve this problem.")

        return aSolvedDetail

    def initial_variables(self, problem: LPS) -> LPVariables:
        """初期点の作成

        x,s はともに非負の値になる必要がある
        複数の決定方法によって初期点を作成し, 制約違反とmuのうちの最大値が最小だった初期点を初期点とする
        決定方法について詳細は p122 7.3.1 参照

        Returns:
            LPVariables: 初期点 x, λ, s
        """
        v_m = self._initial_variables_by_Mehrotra(problem)
        v_l = self._initial_variables_by_Lustig(problem)

        def max_residual_mu(v_0: LPVariables, problem: LPS) -> float:
            """制約残渣, mu のうち最大の値を出力"""
            r_b = problem.residual_main_constraint(v_0.x)
            r_c = problem.residual_dual_constraint(v_0.y, v_0.s)
            return max(np.linalg.norm(r_b), np.linalg.norm(r_c), v_0.mu)

        max_residual_mu_m = max_residual_mu(v_m, problem)
        max_residual_mu_l = max_residual_mu(v_l, problem)
        if max_residual_mu_m < max_residual_mu_l:
            return v_m
        else:
            return v_l

    def _initial_variables_by_Mehrotra(self, problem: LPS) -> LPVariables:
        """Mehrotra の `On the implementation of a primal-dual interior point method` を
        参考にした初期点作成
        """
        A = problem.A
        c = problem.c

        AA_T_inv = np.linalg.inv(A.dot(A.T))
        yhat = AA_T_inv.dot(A).dot(c)
        s_hat = c - A.T.dot(yhat)
        x_hat = A.T.dot(AA_T_inv).dot(problem.b)

        delta_x = max([-1.1 * min(x_hat), 0])
        delta_s = max([-1.1 * min(s_hat), 0])
        x_delta = x_hat + delta_x
        s_delta = s_hat + delta_s

        delta_hat_x = delta_x
        if sum(s_delta) != 0:
            delta_hat_x += 0.5 * (x_delta).T.dot(s_delta) / sum(s_delta)
        delta_hat_s = delta_s
        if sum(x_delta) != 0:
            delta_hat_s += 0.5 * (x_delta).T.dot(s_delta) / sum(x_delta)

        output = LPVariables(
            x_hat + delta_hat_x, yhat, s_hat + delta_hat_s
        )
        return output

    def _initial_variables_by_Lustig(self, problem: LPS) -> LPVariables:
        """Lustig etc. の
        `On implementing Mehrotra's predictor corrector interior-point method
        for linear programming` を参考にした初期点作成
        """
        A = problem.A
        b = problem.b
        c = problem.c

        AA_T_inv = np.linalg.inv(A.dot(A.T))
        x_hat = A.T.dot(AA_T_inv).dot(b)
        xi_1 = max(-100 * x_hat.min(), 100, np.linalg.norm(b, ord=1) / 100)
        xi_2 = 1 + np.linalg.norm(c, ord=1)
        x_0 = x_hat.copy()
        s_0 = np.tile(xi_2, problem.n)
        for i in range(problem.n):
            x_0[i] = max(x_0[i], xi_1)
            if c[i] > xi_2 or (c[i] >= 0 and c[i] < xi_2):
                s_0[i] = c[i] + xi_2
            elif c[i] < -xi_2:
                s_0[i] = -c[i]
        return LPVariables(x_0, np.zeros(problem.m), s_0)

    def _initial_variables_constant(self, problem: LPS) -> LPVariables:
        """定数で与えられた初期点の作成

        (x,y,s) = (C,0,C) となる

        Returns:
            LPVariables: 初期点 x, λ, s
        """
        return LPVariables(np.ones(problem.n) * self.initial_point_scale, np.zeros(problem.m), np.ones(problem.n) * self.initial_point_scale)

    def log_initial_problem_information(self, problem_0: LPS):
        """最初の段階での問題に関するロギングの実行"""
        logger.info(f"{indent}Dimension of problem n: {problem_0.n}, m: {problem_0.m}")

        logger.info(f"{indent}Eigen values:")
        logger.info(f"{indent*2}Max: {problem_0.max_sqrt_eigen_value_AAT}, Min: {problem_0.min_sqrt_eigen_value_AAT}")

        logger.info(f"{indent}Abs ratio scaling:")
        logger.info(f"{indent*2}Max: {problem_0.max_abs_A}, Min: {problem_0.min_abs_A_nonzero}")

        logger.info(f"{indent}Condition number: {problem_0.condition_number_A}")
        if not problem_0.is_full_row_rank():
            rank = np.linalg.matrix_rank(problem_0.A)
            logger.warning(f"{indent}Constraint matrix is not full row rank! m: {problem_0.m}, rank: {rank}")

    def log_initial_situation(self, v_0: LPVariables, problem_0: LPS):
        """最初の段階での変数に関するロギングの実行"""
        self.log_positive_variables_negativity(v_0)
        logger.info(f"{indent}Objective function: {problem_0.objective_main(v_0.x):.2f}")

        logger.info(f"{indent}Duality parameter: {v_0.mu}")

        logger.info(f"{indent}Max constraint violation:")
        logger.info(f"{indent*2}main: {np.linalg.norm(problem_0.residual_main_constraint(v_0.x), np.inf)}")
        logger.info(f"{indent*2}dual: {np.linalg.norm(problem_0.residual_dual_constraint(v_0.y, v_0.s), np.inf)}")

    def log_positive_variables_negativity(self, v: LPVariables):
        """もし x, s が負になってしまった場合アルゴリズムが狂うので, 負になっていないか確認"""
        logger.info("Check x and s negativity.")
        min_x = min(v.x)
        min_s = min(v.s)
        logger.info(f"{indent}min x: {min_x}, min s: {min_s}")
        if min_x < 0:
            idx_ = np.where(v.x < 0)[0]
            logger.warning(f"x is negative! Negative index: {idx_}")
        if min_s < 0:
            idx_ = np.where(v.s < 0)[0]
            logger.warning(f"s is negative! Negative index: {idx_}")

    @abstractmethod
    def run_algorithm(self, problem_0: LPS, v_0: LPVariables) -> SolvedDetail:
        """反復で解くアルゴリズム部分の実行

        Returns:
            SolvedDetail: 最適解に関する詳細情報を格納したデータ構造
        """
        pass

    def scale_step_size(self, alpha: float, iter_num: int) -> float:
        """ステップサイズが次の反復点が内点であることを保証するためのスケーリング"""
        # beta = 1 - np.exp(-(iter_num + 2))
        beta = 0.9
        return beta * alpha

    def is_calculation_time_reached_upper(self, elapsed_time: float) -> bool:
        """計算時間が上限に達したか.
        もし設定が 0以下の値であれば, 時間無制限とする

        Args:
            elapsed_time (float): 経過秒数. `time.time()` で得られる秒数を基準
        """
        upper_bound = self.parameters.CALC_TIME_UPPER
        if upper_bound <= 0:
            return False
        return elapsed_time > upper_bound

    def make_SolvedSummary(
        self, v: LPVariables, problem: LPS,
        is_solved: bool,
        iter_num: int, is_iteration_number_reached_upper: bool,
        elapsed_time: float,
    ) -> SolvedSummary:
        """最適化の結果の概要を出力する"""
        output = SolvedSummary(
            problem.name, self.__class__.__name__, self.config_section,
            False,
            problem.n, problem.m,
            is_solved,
            iter_num,
            # 反復回数上限に達し, それでもまだ解けてない場合に反復を追加しようとするので over upper
            is_iteration_number_reached_upper and not is_solved,
            round(elapsed_time, 2),
            self.is_calculation_time_reached_upper(elapsed_time) and not is_solved,
            problem.objective_main(v.x), v.mu,
            np.linalg.norm(problem.residual_main_constraint(v.x), np.inf),
            np.linalg.norm(problem.residual_dual_constraint(v.y, v.s), np.inf),
        )
        return output

    def log_solved_data(self, aSolvedDetail: SolvedDetail):
        """最適化の結果をロギング.
        他のソルバーでも使用することが考えられるので関数に

        Args:
            aSolvedDetail (SolvedDetail): 求解結果
        """
        aSolvedSummary = aSolvedDetail.aSolvedSummary

        logger.info(f"Needed iteration: {aSolvedSummary.iter_num}")
        logger.debug("Dimension of solved problem:")
        logger.debug(f"{indent}n: {aSolvedDetail.problem.n}, m: {aSolvedDetail.problem.m}")
        logger.info(f"Objective function: {aSolvedSummary.obj:.2f}")
        logger.info(f"Duality parameter: {aSolvedSummary.mu}")
        logger.info("Max constraint violation:")
        logger.info(f"{indent}main: {aSolvedSummary.max_r_b}")
        logger.info(f"{indent}dual: {aSolvedSummary.max_r_c}")
        self.log_positive_variables_negativity(aSolvedDetail.v)
