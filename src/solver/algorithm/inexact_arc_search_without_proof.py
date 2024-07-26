import numpy as np

from ...problem import LinearProgrammingProblemStandard as LPS
from ..variables import LPVariables
from .inexact_interior_point_method import InexactArcSearchIPM


class InexactArcSearchIPMWithoutProof(InexactArcSearchIPM):
    """inexact arc-search を基本として, 理論的証明はないが数値実験的に良くなるであろう実装を施したクラス"""

    def decide_step_size(
        self,
        v: LPVariables,
        problem: LPS,
        gamma_2: float,
        x_dot: np.ndarray,
        y_dot: np.ndarray,
        s_dot: np.ndarray,
        x_ddot: np.ndarray,
        y_ddot: np.ndarray,
        s_ddot: np.ndarray,
    ) -> float:
        """step size を決定.
        近傍に入る step size になるまで Armijo のルールに従う.
        pi/2 から確認をはじめ, alpha が近傍に入らなかった場合, pi - alpha の場合も入るか確認する（sin(alpha)=sin(pi - alpha) であるため, 選択肢が増えた分収束も早くなるはず）

        Returns:
            float: step size
        """
        alpha = np.pi / 2

        def is_x_s_positive_and_v_in_neighborhood(v_alpha: LPVariables, alpha: float) -> bool:
            if min(v_alpha.x) < 0 or min(v_alpha.s) < 0:
                return False
            if not self.is_in_center_path_neighborhood(v_alpha, problem, gamma_2):
                return False
            if not self.is_G_and_g_no_less_than_0(v_alpha, v, alpha):
                return False
            return True

        while alpha > self.min_step_size:
            v_alpha = LPVariables(
                self.variable_updater.run(v.x, x_dot, x_ddot, alpha),
                self.variable_updater.run(v.y, y_dot, y_ddot, alpha),
                self.variable_updater.run(v.s, s_dot, s_ddot, alpha),
            )
            if is_x_s_positive_and_v_in_neighborhood(v_alpha, alpha) and self.is_h_no_less_than_0(v_alpha, v, alpha):
                break

            # pi - alpha とした場合
            pi_minus_alpha = np.pi - alpha
            v_alpha = LPVariables(
                self.variable_updater.run(v.x, x_dot, x_ddot, pi_minus_alpha),
                self.variable_updater.run(v.y, y_dot, y_ddot, pi_minus_alpha),
                self.variable_updater.run(v.s, s_dot, s_ddot, pi_minus_alpha),
            )
            if is_x_s_positive_and_v_in_neighborhood(v_alpha, pi_minus_alpha) and self.is_h_no_less_than_0(
                v_alpha, v, pi_minus_alpha
            ):
                alpha = pi_minus_alpha
                break
            alpha *= 3 / 4
        return alpha
