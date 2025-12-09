from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel

from ..problem import LinearProgrammingProblemStandard
from .variables import LPVariables


class SolvedSummary(BaseModel):
    """最適化された際の結果を概要でまとめるクラス

    csv に書き込みすることができるように, list などarray型の要素は入れない.
    入れる場合は `SolvedDetail` クラスへ

    Attributes:
        problem_name: 問題名
        solver_name: 使用したソルバーの名前
        config_section: 使用したパラメータのセクション名
        is_error: エラーが起きた問題かどうか
        n: 問題の変数次元数
        m: 問題の制約数
        num_nonzero: 問題の制約行列の非ゼロ数
        is_solved: 最適化されたか否か
        iter_num: 反復回数
        is_iter_over_upper: 反復回数上限に達したか
        elapsed_time: アルゴリズムが動いていた時間
        is_calc_time_over_upper: 計算時間が上限を超えたか
        obj: 目的関数値
        mu: duality measure or barrier function coef. この値が0に近ければ最適解
        max_r_b: 主問題の制約に関する infeasiblity. 絶対値の最大値
        max_r_c: 双対問題の制約に関する infeasiblity. 絶対値の最大値
    """

    problem_name: str
    solver_name: str
    config_section: str
    is_error: bool
    n: int
    m: int
    num_nonzero: int
    is_solved: bool
    iter_num: int | None = None
    is_iter_over_upper: bool | None = None
    elapsed_time: float | None = None
    is_calc_time_over_upper: bool | None = None
    obj: float | None = None
    mu: float | None = None
    max_r_b: float | None = None
    max_r_c: float | None = None


@dataclass
class SolvedDetail:
    """最適化された際の結果を詳細にまとめるクラス

    csv に書き込む際に出力が難しいリストなどをこちらにまとめる

    Attributes:
        aSolvedSummary: 結果の概要
        v: 最適解
        problem: 対象となった問題. 変形された場合は変形後の結果が格納される
        v_0: 初期解
        problem_0: 初期問題. 変形される前が格納される
        lst_variables_by_iter: 各反復ごとにとられた変数. 使用する変数をすべてまとめている
        lst_merit_func_by_iter: 各反復ごとの merit function の値をまとめたリスト
        lst_main_step_size_by_iter: 各反復ごとの主変数に関する step size の値をまとめたリスト
        lst_dual_step_size_by_iter: 各反復ごとの双対変数に関する step size の値をまとめたリスト
        lst_mu_by_iter: 各反復ごとの mu をまとめたリスト
        lst_norm_vdot_by_iter: 各反復ごとの一階微分のノルムをまとめたリスト
        lst_norm_vddot_by_iter: 各反復ごとの二階微分のノルムをまとめたリスト
        lst_max_norm_main_constraint_by_iter: 各反復ごとの主問題制約残差の最大値をまとめたリスト
        lst_max_norm_dual_constraint_by_iter: 各反復ごとの双対問題制約残差の最大値をまとめたリスト
        lst_iteration_number_updated_by_iterative_refinement: iterative refinement によって問題が更新された際の反復
        lst_residual_inexact_vdot: inexact に一階微分を求めた際の誤差
        lst_tolerance_inexact_vdot: inexact に一階微分を求める際の誤差許容度
        lst_residual_inexact_vddot: inexact に二階微分を求めた際の誤差
        lst_tolerance_inexact_vddot: inexact に二階微分を求める際の誤差許容度
    """

    aSolvedSummary: SolvedSummary
    v: LPVariables | None
    problem: LinearProgrammingProblemStandard
    v_0: LPVariables
    problem_0: LinearProgrammingProblemStandard
    lst_variables_by_iter: list[LPVariables] = field(default_factory=list)
    lst_merit_function_by_iter: list[float] = field(default_factory=list)
    lst_main_step_size_by_iter: list[float] = field(default_factory=list)
    lst_dual_step_size_by_iter: list[float] = field(default_factory=list)
    lst_mu_by_iter: list[float] = field(default_factory=list)
    lst_norm_vdot_by_iter: list[float] = field(default_factory=list)
    lst_norm_vddot_by_iter: list[float] = field(default_factory=list)
    lst_max_norm_main_constraint_by_iter: list[float] = field(default_factory=list)
    lst_max_norm_dual_constraint_by_iter: list[float] = field(default_factory=list)
    lst_iteration_number_updated_by_iterative_refinement: list[float] = field(default_factory=list)
    lst_residual_inexact_vdot: list[float] = field(default_factory=list)
    lst_tolerance_inexact_vdot: list[float] = field(default_factory=list)
    lst_residual_inexact_vddot: list[float] = field(default_factory=list)
    lst_tolerance_inexact_vddot: list[float] = field(default_factory=list)

    @property
    def mu_0(self):
        """問題が解かれる前 mu."""
        return self.v_0.mu

    @property
    def max_r_b_0(self):
        """主問題の制約に関する, 問題が解かれる前の infeasibility"""
        return (np.linalg.norm(self.problem_0.residual_main_constraint(self.v_0.x), np.inf),)

    @property
    def max_r_c_0(self):
        """双対問題の制約に関する, 問題が解かれる前の infeasibility"""
        return np.linalg.norm(self.problem_0.residual_dual_constraint(self.v_0.y, self.v_0.s), np.inf)
