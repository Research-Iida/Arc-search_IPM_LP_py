"""solver に関する module

今後ソルバーの設定を変えることで実験することが頻繁に起こるため,
インターフェイスを使用して変更に対して柔軟な設計をできるようにしておく
"""

from ..logger import get_main_logger, indent
from ..problem import LinearProgrammingProblemStandard as LPS
from .algorithm.algorithm import ILPSolvingAlgoritm
from .optimization_parameters import OptimizationParameters
from .solved_checker import AbsoluteSolvedChecker, RelativeSolvedChecker, SolvedChecker
from .solved_data import SolvedDetail, SolvedSummary
from .variables import LPVariables

logger = get_main_logger()


class LPSolver:
    """LPを解くためのソルバーに関する抽象クラス"""

    solved_checker: SolvedChecker
    # TODO: solver_config_section に改名
    config_section: str
    parameters: OptimizationParameters
    algorithm: ILPSolvingAlgoritm

    def _set_config_and_parameters(self, config_section: str):
        self.config_section = config_section
        self.parameters = OptimizationParameters.import_(config_section)

    def __init__(
        self,
        config_section: str,
        algorithm: ILPSolvingAlgoritm,
        solved_checker: SolvedChecker | None = None,
    ):
        """インスタンス初期化

        Args:
            config_section (str): 設定ファイルのセクション名.
                logging にも使用するので文字列で取得しておく
        """
        self._set_config_and_parameters(config_section)
        self.algorithm = algorithm

        # TODO: ややこしい設定は builder クラスへ委譲
        if solved_checker is None:
            threshold = self.parameters.STOP_CRITERIA_PARAMETER
            if self.parameters.IS_STOPPING_CRITERIA_RELATIVE:
                self.solved_checker = RelativeSolvedChecker(threshold, self.parameters.THRESHOLD_XS_NEGATIVE)
            else:
                self.solved_checker = AbsoluteSolvedChecker(threshold, self.parameters.THRESHOLD_XS_NEGATIVE)
        else:
            self.solved_checker = solved_checker

    def run(self, problem: LPS, v_0: LPVariables | None = None) -> SolvedDetail:
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

        # アルゴリズムの実行
        try:
            aSolvedDetail = self.algorithm.run(problem, v_0)

            logger.info(f"[{solver_name}] [{self.config_section}] End solving {problem.name}.")
            self.log_solved_data(aSolvedDetail)
        # 計算上でエラーが起きても計算が止まらないようにエラー文を生成だけして結果を書き込む
        except Exception as e:
            logger.exception("Error occured - ", exc_info=e)
            aSolvedSummary = SolvedSummary(
                problem.name, solver_name, self.config_section, True, problem.n, problem.m, False
            )
            aSolvedDetail = SolvedDetail(aSolvedSummary, v_0, problem, v_0, problem)

        # 求解不可能だった場合, ログに残す
        if not aSolvedDetail.aSolvedSummary.is_solved:
            logger.warning(f"{solver_name} cannot solve this problem.")

        return aSolvedDetail

    def log_initial_problem_information(self, problem_0: LPS):
        """最初の段階での問題に関するロギングの実行"""
        return self.algorithm.log_initial_problem_information(problem_0)

    def log_positive_variables_negativity(self, v: LPVariables):
        """もし x, s が負になってしまった場合アルゴリズムが狂うので, 負になっていないか確認"""
        return self.algorithm.log_positive_variables_negativity(v)

    def is_calculation_time_reached_upper(self, elapsed_time: float) -> bool:
        """計算時間が上限に達したか.
        もし設定が 0以下の値であれば, 時間無制限とする

        Args:
            elapsed_time (float): 経過秒数. `time.time()` で得られる秒数を基準
        """
        return self.algorithm.is_calculation_time_reached_upper(elapsed_time)

    def make_SolvedSummary(
        self,
        v: LPVariables,
        problem: LPS,
        is_solved: bool,
        iter_num: int,
        is_iteration_number_reached_upper: bool,
        elapsed_time: float,
    ) -> SolvedSummary:
        """最適化の結果の概要を出力する"""
        return self.algorithm.make_SolvedSummary(
            v, problem, is_solved, iter_num, is_iteration_number_reached_upper, elapsed_time
        )

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
