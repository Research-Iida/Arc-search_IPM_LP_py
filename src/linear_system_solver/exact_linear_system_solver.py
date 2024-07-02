"""
探索方向における線形方程式を解くクラス.
Strategy パターン採用.

アルゴリズムごとに別の線形方程式ソルバーを使用できるようにクラス化
"""

from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse import csr_matrix as Csr

from ..logger import get_main_logger

logger = get_main_logger()


class AbstractLinearSystemSolver(metaclass=ABCMeta):
    """線形方程式を解く抽象クラス

    Attributes:
        prev_A: 前回使用した係数行列. `solve` する際に同じものが入力された場合,
            再び同じ前処理をしなくて済むように持っておく
    """

    prev_A: Csr = None

    @abstractmethod
    def solve(self, A: Csr, b: np.ndarray, tolerance: float | None, *args) -> np.ndarray:
        """線形方程式 Ax=b を解き, x を求める

        Args:
            A (Csr): 係数行列
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 出した解が厳密解とどれだけ離れるかの許容度

        Returns:
            np.ndarray: 解
        """
        pass


class ExactLinearSystemSolver(AbstractLinearSystemSolver):
    """線形方程式を正確に解くクラス
    シンプルに numpy で解く

    Attributes:
        prev_lu_factor: `prev_A` を `scipy.linalg.lu_factor` にかけて求めたLU分解.
    """

    prev_lu_factor = None

    def solve(self, A: Csr, b: np.ndarray, tolerance: float | None = None) -> np.ndarray:
        """線形方程式 Ax=b を numpy によるLU分解によって解く"""
        if self.prev_lu_factor is not None:
            if self.prev_A.shape == A.shape and len((self.prev_A != A).data) == 0:
                logger.info("Use prev_A information.")
                return lu_solve(self.prev_lu_factor, b)

        LU_factor = lu_factor(A)

        self.prev_A = A.copy()
        self.prev_lu_factor = LU_factor

        return lu_solve(LU_factor, b)
