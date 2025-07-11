from __future__ import annotations

import numpy as np
from dataclasses import dataclass


class VariablesDimensionMismatchError(Exception):
    pass


@dataclass
class LPVariables:
    """LP にて使用される変数をまとめたクラス

    Attributes:
        x: 主問題の変数 x
        y: 制約にかかるラグランジュ係数 λ
        s: 双対問題の変数 s
    """

    x: np.ndarray
    y: np.ndarray
    s: np.ndarray

    def __post_init__(self):
        if self.x.shape != self.s.shape:
            raise VariablesDimensionMismatchError("x and s must have the same shape")

    @property
    def mu(self) -> float:
        """双対パラメータμを取得する"""
        return np.dot(self.x, self.s) / self.x.shape[0]

    def __add__(self, addend: LPVariables) -> LPVariables:
        if self.x.shape != addend.x.shape:
            raise VariablesDimensionMismatchError("x の次元が合いません.")
        if self.y.shape != addend.y.shape:
            raise VariablesDimensionMismatchError("y の次元が合いません.")
        if self.s.shape != addend.s.shape:
            raise VariablesDimensionMismatchError("s の次元が合いません.")

        return LPVariables(self.x + addend.x, self.y + addend.y, self.s + addend.s)

    def __mul__(self, multiplier: float) -> LPVariables:
        return LPVariables(multiplier * self.x, multiplier * self.y, multiplier * self.s)

    def remove_constraint_rows(self, remove_rows: list[int]) -> LPVariables:
        """制約の行が減るのに合わせて変数を削除する

        Args:
            remove_rows (list[int]): 削除対象の制約の行

        Returns:
            LPVariables: y の対象 idx を削除した変数インスタンス
        """
        return LPVariables(self.x, np.delete(self.y, remove_rows), self.s)

    def insert_constraint_rows(self, insert_rows: list[int]) -> LPVariables:
        """制約の行が増えるのに合わせて変数を追加する

        Args:
            insert_rows (list[int]): 挿入対象の制約の行

        Returns:
            LPVariables: y の対象 idx 分追加した変数インスタンス.
                insert_rows は0, それ以外は insert された分ずれて同じ値を格納
        """
        size_new_y = self.y.shape[0] + len(insert_rows)
        y_inserted = np.zeros(size_new_y)
        idx_not_change = np.ones(size_new_y, dtype=bool)
        idx_not_change[insert_rows] = False
        y_inserted[idx_not_change] = self.y
        return LPVariables(self.x, y_inserted, self.s)

    def isclose(self, other: LPVariables, threshold: float = 10 ** (-6)) -> bool:
        """変数同士で比較して近いか確認する

        Args:
            other (LPVariables): もう片方の変数

        Returns:
            bool: 近い値であれば True
        """
        is_x_close = np.all(np.isclose(self.x, other.x, atol=threshold))
        is_y_close = np.all(np.isclose(self.y, other.y, atol=threshold))
        is_s_close = np.all(np.isclose(self.s, other.s, atol=threshold))
        return is_x_close and is_y_close and is_s_close
