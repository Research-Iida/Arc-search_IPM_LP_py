"""変数の更新方法について管理するモジュール
arc-search, line-search の2通り
"""
from abc import ABCMeta, abstractmethod

import numpy as np


class VariableUpdater(metaclass=ABCMeta):
    max_step_size: float

    def __init__(self, delta_xs: float):
        """初期化

        Args:
            delta_xs (float): x, もしくは s の正値性を保証するため, どの程度まで0より大きくとるか
        """
        self._delta_xs = delta_xs

    def is_max_step_size(self, step_size: float) -> bool:
        return step_size == self.max_step_size

    def max_alpha_guarantee_positive_with_line(self, xs: np.ndarray, xs_dot: np.ndarray) -> float:
        """x-x_dot, s-s_dot が0以上となることを保証する最大のステップサイズの取得

        Args:
            xs: x, もしくは s
            xs_dot: x, もしくは s の一次微分値

        Return:
            float: `xs - alpha * xs_dot` のすべての値が0以上となる最大の `alpha`
        """
        # 一次微分が0より大きい場合のみ alpha の出力対象となる
        xs_dot_positive = xs_dot[xs_dot > 0]
        # もし alpha の出力対象がない場合は1を返す
        if len(xs_dot_positive) == 0:
            return 1

        # xs - xs_dot のすべての要素が0以上となる最大のalphaの出力
        xs_positive = xs[xs_dot > 0]
        output = min(xs_positive / xs_dot_positive)
        return min([output, 1])

    @abstractmethod
    def run(self, v: np.ndarray, v_dot: np.ndarray, v_ddot: np.ndarray, alpha: float) -> np.ndarray:
        """変数の更新の実行

        Args:
            v: x, s, もしくは λ
            v_dot: x, s, もしくはλの一次微分
            v_ddot: x, s, もしくはλの二次微分
            alpha: 対応する step size
        """
        pass

    @abstractmethod
    def max_step_size_guarantee_positive(
        self, xs: np.ndarray, xs_dot: np.ndarray, xs_ddot: np.ndarray
    ) -> float:
        pass


class LineVariableUpdater(VariableUpdater):
    max_step_size: float = 1

    def max_step_size_guarantee_positive(
        self, xs: np.ndarray, xs_dot: np.ndarray, xs_ddot: np.ndarray
    ) -> float:
        """line search の更新で x, s が非負になる最大のステップサイズの取得

        xs は負の値になってはいけないため, 少量分の割合を差し引いてからstep size を決定する

        Args:
            xs: x, もしくは s
            xs_dot: x, もしくは s の一次微分値
            xs_ddot: x, もしくは s の二次微分値

        Return:
            float: `xs - alpha * (xs_dot - xs_ddot)` のすべての値が0以上となる最大の `alpha`
        """
        # （一次微分 - 二次微分）が0より大きい場合のみ alpha の出力対象となる
        xs_dot_minus_xs_ddot = xs_dot - xs_ddot
        target_positive = xs_dot_minus_xs_ddot[xs_dot_minus_xs_ddot > 0]
        # もし alpha の出力対象がない場合は1を返す
        if len(target_positive) == 0:
            return self.max_step_size

        # すべての要素が0以上となる最大のalphaの出力
        xs_positive = xs[xs_dot_minus_xs_ddot > 0] * (1 - self._delta_xs)
        output = min(xs_positive / target_positive)
        return min([output, self.max_step_size])

    def run(self, v: np.ndarray, v_dot: np.ndarray, v_ddot: np.ndarray, alpha: float) -> np.ndarray:
        """line search に従って変数の更新

        Args:
            v: x, s, もしくは λ
            v_dot: x, s, もしくはλの一次微分
            v_ddot: x, s, もしくはλの二次微分
            alpha: 対応する step size
        """
        return v - alpha * (v_dot - v_ddot)


class ArcVariableUpdater(VariableUpdater):
    max_step_size: float = np.pi / 2

    def max_step_size_guarantee_positive(
        self, xs: np.ndarray, xs_dot: np.ndarray, xs_ddot: np.ndarray
    ) -> float:
        """arc-search の更新で x, s が非負になる最大の alpha の出力

        xs は負の値になってはいけないため, 少量分の割合を差し引いてからstep size を決定する
        """
        # 出力の最大値は pi/2. 変数の各要素に応じてこの値が変化
        alpha = self.max_step_size
        xs_pos = xs * (1 - self._delta_xs)
        # Case 番号は論文参照
        for xs_i, xs_dot_i, xs_ddot_i in zip(xs_pos, xs_dot, xs_ddot):
            # Case 1. 一次微分が0, 二次微分が0でない場合
            if xs_dot_i == 0:
                if (xs_plus_xs_ddot := (xs_i + xs_ddot_i)) < 0:
                    alpha = min(alpha, np.arccos(xs_plus_xs_ddot / xs_ddot_i))
            # Case 2. 二次微分が0, 一次微分が0でない場合
            if xs_ddot_i == 0:
                if xs_dot_i > xs_i:
                    alpha = min(alpha, np.arcsin(xs_i / xs_dot_i))
            # Case 3,4. 一次微分が正の場合
            if xs_dot_i > 0:
                denominator = np.sqrt(xs_dot_i**2 + xs_ddot_i**2)
                if (xs_plus_xs_ddot := xs_i + xs_ddot_i) < denominator:
                    alpha_pos = np.arcsin(xs_plus_xs_ddot / denominator)
                    # Case 3. 一次微分, 二次微分がともに正の場合
                    if xs_ddot_i > 0:
                        alpha_neg = np.arcsin(xs_ddot_i / denominator)
                        alpha = min(alpha, alpha_pos - alpha_neg)
                    # Case 4. 一次微分が正, 二次微分が負の場合
                    else:
                        alpha_neg = np.arcsin(-xs_ddot_i / denominator)
                        alpha = min(alpha, alpha_pos + alpha_neg)
            # Case 5. 一次微分, 二次微分がともに負の場合
            if xs_dot_i < 0 and xs_ddot_i < 0:
                denominator = np.sqrt(xs_dot_i**2 + xs_ddot_i**2)
                if abs(xs_plus_xs_ddot := xs_i + xs_ddot_i) < denominator:
                    alpha_1 = np.arcsin(-xs_plus_xs_ddot / denominator)
                    alpha_2 = np.arcsin(-xs_ddot_i / denominator)
                    alpha = min(alpha, np.pi - alpha_1 - alpha_2)
            # Case 6,7 については π/2 になるため, 計算する必要がない
        return alpha

    def run(self, v: np.ndarray, v_dot: np.ndarray, v_ddot: np.ndarray, alpha: float) -> np.ndarray:
        """arc search に従った変数の更新

        Args:
            v: x, s, もしくは λ
            v_dot: x, s, もしくはλの一次微分
            v_ddot: x, s, もしくはλの二次微分
            alpha: 対応する step size
        """
        return v - v_dot * np.sin(alpha) + v_ddot * (1 - np.cos(alpha))
