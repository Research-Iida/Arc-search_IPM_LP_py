"""solver に関する module

今後ソルバーの設定を変えることで実験することが頻繁に起こるため,
インターフェイスを使用して変更に対して柔軟な設計をできるようにしておく
"""

from ..logger import get_main_logger, indent
from ..problem import LinearProgrammingProblemStandard as LPS
from .algorithm.algorithm import ILPSolvingAlgoritm
from .solved_data import SolvedDetail, SolvedSummary
from .variables import LPVariables

logger = get_main_logger()


class LPSolver:
    """LPを解くためのソルバーに関する抽象クラス"""

    algorithm: ILPSolvingAlgoritm

    def __init__(
        self,
        algorithm: ILPSolvingAlgoritm,
    ):
        """インスタンス初期化. アルゴリズムを設定する"""
        self.algorithm = algorithm

    @property
    def algorithm_config_section(self) -> str:
        """algorithm の config_section. logging でたびたび使用.

        Returns:
            str: `algorithm.config_section`
        """
        return self.algorithm.config_section

    def run(self, problem: LPS, v_0: LPVariables | None = None) -> SolvedDetail:
        """入力されたLPに対してアルゴリズムを実行

        Aの行数とbの次元数, およびAの列数とcの次元数が異なる場合, エラーを起こす

        Args:
            problem: LPS における係数群をまとめたクラスインスタンス
            v_0: 初期点

        Returns:
            SolvedDetail: 最適解に関する情報をまとめたインスタンス
        """
        algorithm_name = self.algorithm.__class__.__name__
        algorithm_config_section = self.algorithm.config_section
        logger.info(f"[{algorithm_name}] [{algorithm_config_section}] Start solving {problem.name}.")

        logger.info("Logging problem information.")
        self.algorithm.log_initial_problem_information(problem)

        # アルゴリズムの実行
        try:
            aSolvedDetail = self.algorithm.run(problem, v_0)

            logger.info(f"[{algorithm_name}] [{algorithm_config_section}] End solving {problem.name}.")
            self.log_solved_data(aSolvedDetail)
        # 計算上でエラーが起きても計算が止まらないようにエラー文を生成だけして結果を書き込む
        except Exception as e:
            logger.exception("Error occured - ", exc_info=e)
            aSolvedSummary = SolvedSummary(
                problem.name, algorithm_name, algorithm_config_section, True, problem.n, problem.m, False
            )
            aSolvedDetail = SolvedDetail(aSolvedSummary, v_0, problem, v_0, problem)

        # 求解不可能だった場合, ログに残す
        if not aSolvedDetail.aSolvedSummary.is_solved:
            logger.warning(f"{algorithm_name} cannot solve this problem.")

        return aSolvedDetail

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
        self.algorithm.log_positive_variables_negativity(aSolvedDetail.v)
