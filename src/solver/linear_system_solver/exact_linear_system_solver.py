"""
探索方向における線形方程式を解くクラス.
Strategy パターン採用.

アルゴリズムごとに別の線形方程式ソルバーを使用できるようにクラス化
"""

from abc import ABCMeta, abstractmethod
from typing import Callable

import numpy as np
from scipy.sparse import csr_matrix as CsrMatrix
from scipy.sparse.linalg import factorized

from ...logger import get_main_logger

logger = get_main_logger()


class AbstractLinearSystemSolver(metaclass=ABCMeta):
    """線形方程式を解く抽象クラス

    Attributes:
        prev_A: 前回使用した係数行列. `solve` する際に同じものが入力された場合,
            再び同じ前処理をしなくて済むように持っておく
    """

    prev_A: CsrMatrix | None = None

    @abstractmethod
    def solve(self, A: CsrMatrix, b: np.ndarray, tolerance: float | None = 10**-7, *args) -> np.ndarray:
        """線形方程式 Ax=b を解き, x を求める

        Args:
            A (CsrMatrix): 係数行列
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 出した解が厳密解とどれだけ離れるかの許容度

        Returns:
            np.ndarray: 解

        TODO:
            * tolerance は必要なクラスとそうでないクラスがあるので, Attribute に移す
        """
        pass


class ExactLinearSystemSolver(AbstractLinearSystemSolver):
    """線形方程式を正確に解くクラス
    シンプルにLU分解で解く

    Attributes:
        prev_factorized: `prev_A` をLU分解した結果. 前回使ったものと同じであれば同様のものを使いまわせる
    """

    prev_factorized: Callable | None = None

    def solve(self, A: CsrMatrix, b: np.ndarray, tolerance: float | None = None) -> np.ndarray:
        """線形方程式 Ax=b を numpy によるLU分解によって解く"""
        if self.prev_factorized is not None:
            if self.prev_A.shape == A.shape and (self.prev_A - A).nnz == 0:
                logger.info("Use prev_A information.")
                return self.prev_factorized(b)

        self.prev_A = A.copy()
        # scipy.sparse.linalg.factorized は csc_matrix を引数に取らないと warning を出す
        self.prev_factorized = factorized(A.tocsc())

        return self.prev_factorized(b)
