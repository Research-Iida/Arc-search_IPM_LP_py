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

    def create_solved_summary(
        self,
        problem: LPS,
        is_error: bool,
        is_solved: bool,
        iter_num: int | None = None,
        elapsed_time: float | None = None,
        v_star: LPVariables | None = None,
        is_iter_over_upper: bool | None = None,
        is_calc_time_over_upper: bool | None = None,
        **kwargs,
    ) -> SolvedSummary:
        """最適化の結果を作成する. 同じクラスから複数の状態を取得できるので, これを基準として作っていく

        Args:
            problem (LPS): 対象の最適化問題
            v_star (LPVariables): solver が実行した結果得られる解
        """
        # 求まっていて SolvedSummary に含めることができるものは追加する
        if iter_num is not None:
            kwargs["iter_num"] = iter_num
        if elapsed_time is not None:
            kwargs["elapsed_time"] = round(elapsed_time, 2)
        if v_star is not None:
            kwargs["mu"] = v_star.mu
            kwargs["obj"] = problem.objective_main(v_star.x)
            kwargs["max_r_b"] = np.linalg.norm(problem.residual_main_constraint(v_star.x), np.inf)
            kwargs["max_r_c"] = np.linalg.norm(problem.residual_dual_constraint(v_star.y, v_star.s), np.inf)
        if is_iter_over_upper is not None:
            kwargs["is_iter_over_upper"] = is_iter_over_upper
        if is_calc_time_over_upper is not None:
            kwargs["is_calc_time_over_upper"] = is_calc_time_over_upper

        result = SolvedSummary(
            problem_name=problem.name,
            solver_name=self.solver_name,
            config_section=self.solver_config_section,
            is_error=is_error,
            n=problem.n,
            m=problem.m,
            num_nonzero=problem.num_nonzero,
            is_solved=is_solved,
            **kwargs,
        )
        return result

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
            aSolvedSummary = self.create_solved_summary(problem=problem, is_error=True, is_solved=False)
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
