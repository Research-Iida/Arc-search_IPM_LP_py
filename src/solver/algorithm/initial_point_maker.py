import abc

import numpy as np

from ...problem import LinearProgrammingProblemStandard as LPS
from ..variables import LPVariables


class IInitialPointMaker(abc.ABC):
    """初期点の作成に責務を持つクラス"""

    @abc.abstractmethod
    def make_initial_point(self, problem: LPS) -> LPVariables:
        """初期点の作成. 中身はサブクラス参照"""
        pass


class MehrotraInitialPointMaker(IInitialPointMaker):
    """Mehrotra の `On the implementation of a primal-dual interior point method` を
    参考にした初期点作成
    """

    def make_initial_point(self, problem: LPS) -> LPVariables:
        """初期点の作成"""
        A = problem.A
        c = problem.c
        # AA_T_inv = problem.AA_T_inv

        yhat = problem.AA_T_pre_factorized(A @ c)
        s_hat = c - A.T.tocsr() @ yhat
        tmp = problem.AA_T_pre_factorized(problem.b)
        x_hat = A.T.tocsr() @ tmp

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

        output = LPVariables(x_hat + delta_hat_x, yhat, s_hat + delta_hat_s)
        return output


class LustingInitialPointMaker(IInitialPointMaker):
    """Lustig etc. の
    `On implementing Mehrotra's predictor corrector interior-point method for linear programming`
    を参考にした初期点作成
    """

    def make_initial_point(self, problem: LPS) -> LPVariables:
        """初期点の作成"""
        A = problem.A
        b = problem.b
        c = problem.c

        tmp = problem.AA_T_pre_factorized(b)
        x_hat = A.T.tocsr() @ tmp
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


class YangInitialPointMaker(IInitialPointMaker):
    """Yang の
    `CurveLP-A MATLAB implementation of an infeasible interior-point algorithm for linear programming`
    を参考にした初期点作成
    """

    def make_initial_point(self, problem: LPS) -> LPVariables:
        """初期点の作成

        x,s はともに非負の値になる必要がある
        複数の決定方法によって初期点を作成し, 制約違反とmuのうちの最大値が最小だった初期点を初期点とする
        決定方法について詳細は ``Arc-Search Techniques for Interior-Point Methods'' p122 Section 7.3.1 参照

        Returns:
            LPVariables: 初期点 x, y, s
        """
        v_m = MehrotraInitialPointMaker().make_initial_point(problem)
        v_l = LustingInitialPointMaker().make_initial_point(problem)

        def max_residual_mu(v_0: LPVariables, problem: LPS) -> float:
            """制約残渣, mu のうち最大の値を出力"""
            r_b = problem.residual_main_constraint(v_0.x)
            r_c = problem.residual_dual_constraint(v_0.y, v_0.s)
            return max(np.linalg.norm(r_b), np.linalg.norm(r_c), v_0.mu)

        max_residual_mu_m = max_residual_mu(v_m, problem)
        max_residual_mu_l = max_residual_mu(v_l, problem)
        if max_residual_mu_m < max_residual_mu_l:
            return v_m
        return v_l


class ConstantInitialPointMaker(IInitialPointMaker):
    """定数で与えられる初期点の作成. 定数はインスタンス初期化時に決定"""

    def __init__(self, initial_point_scale: int):
        self.initial_point_scale = initial_point_scale

    def make_initial_point(self, problem: LPS) -> LPVariables:
        """初期点の作成"""
        return LPVariables(
            np.ones(problem.n) * self.initial_point_scale,
            np.zeros(problem.m),
            np.ones(problem.n) * self.initial_point_scale,
        )
