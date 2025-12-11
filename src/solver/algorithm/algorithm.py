import abc

import numpy as np

from ...logger import get_main_logger, indent
from ...problem import LinearProgrammingProblemStandard as LPS
from ..optimization_parameters import OptimizationParameters
from ..solved_checker import SolvedChecker
from ..solved_data import SolvedSummary
from ..solver import ILPSolver
from ..variables import LPVariables

logger = get_main_logger()


class ILPSolvingAlgorithm(ILPSolver, metaclass=abc.ABCMeta):
    """LP を解くアルゴリズムのインターフェース. IPM などが実装にあたる"""

    config_section: str
    parameters: OptimizationParameters
    solved_checker: SolvedChecker

    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
        solved_checker: SolvedChecker,
    ):
        """インスタンス初期化

        Args:
            config_section (str): 設定ファイルのセクション名.
                logging にも使用するので文字列で取得しておく
        """
        self.config_section = config_section
        self.parameters = parameters
        self.solved_checker = solved_checker
        logger.info(f"Checking solved by the method of {self.solved_checker.__class__.__name__}")

    @property
    def solver_name(self) -> str:
        return self.__class__.__name__

    @property
    def solver_config_section(self) -> str:
        return self.config_section

    @property
    def min_step_size(self) -> float:
        return self.parameters.MIN_STEP_SIZE

    @property
    def initial_point_scale(self) -> int:
        return self.parameters.INITIAL_POINT_SCALE

    def scale_step_size(self, alpha: float, iter_num: int) -> float:
        """ステップサイズが次の反復点が内点であることを保証するためのスケーリング"""
        # beta = 1 - np.exp(-(iter_num + 2))
        beta = 0.9
        return beta * alpha

    def log_initial_problem_information(self, problem_0: LPS):
        """最初の段階での問題に関するロギングの実行"""
        logger.info(f"{indent}Dimension of problem n: {problem_0.n}, m: {problem_0.m}")

        logger.info(f"{indent}Abs ratio scaling:")
        logger.info(f"{indent*2}Max: {problem_0.max_abs_A}, Min: {problem_0.min_abs_A_nonzero}")

        # 以下は時間がかかるので実行しない
        # TODO: 問題のサイズによって判定させる？
        # logger.info(f"{indent}Eigen values:")
        # logger.info(f"{indent*2}Max: {problem_0.max_sqrt_eigen_value_AAT}, Min: {problem_0.min_sqrt_eigen_value_AAT}")
        # logger.info(f"{indent}Condition number: {problem_0.condition_number_A}")
        # if not problem_0.is_full_row_rank():
        #     logger.warning(
        #         f"{indent}Constraint matrix is not full row rank! m: {problem_0.m}, rank: {problem_0.row_rank}"
        #     )

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

    def log_initial_situation(self, problem_0: LPS, v_0: LPVariables):
        """最初の段階での問題, 変数に関するロギングの実行"""
        logger.info("Logging initial situation.")

        self.log_initial_problem_information(problem_0)
        self.log_positive_variables_negativity(v_0)

        logger.info(f"{indent}Objective function: {problem_0.objective_main(v_0.x):.2f}")

        logger.info(f"{indent}Duality parameter: {v_0.mu}")

        logger.info(f"{indent}Max constraint violation:")
        logger.info(f"{indent*2}main: {np.linalg.norm(problem_0.residual_main_constraint(v_0.x), np.inf)}")
        logger.info(f"{indent*2}dual: {np.linalg.norm(problem_0.residual_dual_constraint(v_0.y, v_0.s), np.inf)}")

    def is_calculation_time_reached_upper(self, elapsed_time: float) -> bool:
        """計算時間が上限に達したか.
        設定がなければ一律 False を返す

        Args:
            elapsed_time (float): 経過秒数. `time.time()` で得られる秒数を基準
        """
        if self.parameters.CALC_TIME_UPPER is None:
            return False

        upper_bound = self.parameters.CALC_TIME_UPPER
        return elapsed_time > upper_bound

    def create_non_error_solved_summary(
        self,
        v: LPVariables,
        problem: LPS,
        is_solved: bool,
        iter_num: int,
        is_iteration_number_reached_upper: bool,
        elapsed_time: float,
    ) -> SolvedSummary:
        """最適化の結果の概要を出力する"""
        output = self.create_solved_summary(
            problem=problem,
            is_error=False,
            is_solved=is_solved,
            iter_num=iter_num,
            elapsed_time=round(elapsed_time, 2),
            v_star=v,
            # 反復回数上限に達し, それでもまだ解けてない場合に反復を追加しようとするので over upper
            is_iter_over_upper=is_iteration_number_reached_upper and not is_solved,
            is_calc_time_over_upper=self.is_calculation_time_reached_upper(elapsed_time) and not is_solved,
        )
        return output
