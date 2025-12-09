"""solver に関する module

今後ソルバーの設定を変えることで実験することが頻繁に起こるため,
インターフェイスを使用して変更に対して柔軟な設計をできるようにしておく
"""

import abc

import numpy as np

from ..logger import get_main_logger, indent
from ..problem import LinearProgrammingProblemStandard as LPS
from .solved_data import SolvedDetail, SolvedSummary
from .variables import LPVariables

logger = get_main_logger()


class ILPSolver(abc.ABC):
    @property
    @abc.abstractmethod
    def solver_name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def solver_config_section(self) -> str:
        pass

    @abc.abstractmethod
    def _execute(self, problem: LPS, v_0: LPVariables | None) -> SolvedDetail:
        """最適化問題を解く. 解き方は具象クラスで定義する"""
        pass

    def run(self, problem: LPS, v_0: LPVariables | None = None) -> SolvedDetail:
        """入力されたLPに対してアルゴリズムを実行

        Aの行数とbの次元数, およびAの列数とcの次元数が異なる場合, エラーを起こす

        Args:
            problem: LPS における係数群をまとめたクラスインスタンス
            v_0: 初期点

        Returns:
            SolvedDetail: 最適解に関する情報をまとめたインスタンス
        """
        msg_prefix = f"[{self.solver_name}] [{self.solver_config_section}]"
        logger.info(f"{msg_prefix} Start solving {problem.name}.")

        # アルゴリズムの実行
        try:
            aSolvedDetail = self._execute(problem, v_0)

            logger.info(f"{msg_prefix} End solving {problem.name}.")
            self.log_solved_data(aSolvedDetail)
        # 計算上でエラーが起きても計算が止まらないようにエラー文を生成だけして結果を書き込む
        except Exception:
            logger.exception("Error occurred - ")
            aSolvedSummary = SolvedSummary(
                problem_name=problem.name,
                solver_name=self.solver_name,
                config_section=self.solver_config_section,
                is_error=True,
                n=problem.n,
                m=problem.m,
                is_solved=False,
            )
            aSolvedDetail = SolvedDetail(aSolvedSummary, v_0, problem, v_0, problem)

        # 求解不可能だった場合, ログに残す
        if not aSolvedDetail.aSolvedSummary.is_solved:
            logger.warning(f"{msg_prefix} Cannot solve {problem.name}.")

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
        self.log_positive_variables_negativity(aSolvedDetail.v)

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
