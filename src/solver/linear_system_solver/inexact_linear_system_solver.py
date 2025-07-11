"""
探索方向における線形方程式を解くクラス.
Strategy パターン採用.

アルゴリズムごとに別の線形方程式ソルバーを使用できるようにクラス化
"""

from abc import ABCMeta

import numpy as np
import scipy.sparse.linalg as spla
from scipy.sparse import csr_matrix as Csr
from scipy.sparse import diags

from ...logger import get_main_logger
from .exact_linear_system_solver import AbstractLinearSystemSolver

logger = get_main_logger()


class AbstractInexactLinearSystemSolver(AbstractLinearSystemSolver, metaclass=ABCMeta):
    """線形方程式を inexact に解くクラスの super class"""

    pass


class CGLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """共役勾配法(Conjugate Gradient)で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """

    prev_coef_matrix_preprocessed: Csr = None
    method_name: str = "Conjugate Gradient method"

    def solve(self, A: Csr, b: np.ndarray, tolerance: float = 10**-7, *args) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (Csr): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        # 数値誤差により係数行列が対称にならない場合があるため, 少し修正
        coef_matrix: Csr = (A + A.T) / 2
        right_hand_side = b.copy()

        # もし A が前に使ったものと同じでなければ前処理を施す
        if self.prev_A is not None and self.prev_A.shape == A.shape:
            if (self.prev_A - A).nnz == 0:
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed

        logger.info(f"{self.method_name} start.")
        result, info = spla.cg(
            coef_matrix,
            right_hand_side,
            rtol=0,
            atol=tolerance,
            M=diags(1 / coef_matrix.diagonal()),
            maxiter=100 * coef_matrix.shape[0],
        )
        logger.info(f"{self.method_name} end.")

        # 理論的には係数行列は半正定値になるはずだが, 誤差の範囲内におさまらない場合, 数値誤差の影響で半正定値でない可能性がある
        # その場合だけ, 最小固有値分摂動を行って再度解く（毎回固有値を求めるのはコストが大きいので回数は少なくしたい）
        # 以下は実行してもあまり意味ないことが数値実験でわかったので, やらないようにする
        # if np.linalg.norm(coef_matrix @ result - right_hand_side) > tolerance:
        #     logger.warning(f"{self.method_name} cannot solve within the tolerance!")
        #     max_eigen_value = spla.eigsh(coef_matrix, k=1, which="LM", return_eigenvectors=False)
        #     if not max_eigen_value > 0:
        #         logger.warning("Coefficient matrix of the LSS is not positive definite!")
        #         coef_perturbation = 10 ** (-3)
        #         logger.info(f"Perturbation coefficient matrix by {coef_perturbation}.")
        #         coef_matrix += coef_perturbation * eye(coef_matrix.shape[0])

        #         result, info = spla.cg(
        #             coef_matrix,
        #             right_hand_side,
        #             tol=0,
        #             atol=tolerance,
        #             M=diags(1 / coef_matrix.diagonal()),
        #             # maxiter=100 * coef_matrix.shape[0],
        #         )

        # cg法が解けなかった場合の warning
        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        # 疎行列にして保管しておく
        self.prev_coef_matrix_preprocessed = Csr(coef_matrix)
        return result


class BiCGLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """双共役勾配法で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """

    prev_coef_matrix_preprocessed: Csr = None
    method_name: str = "BiCG"

    def solve(self, A: np.ndarray, b: np.ndarray, tolerance: float = 10**-7, *args) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        # もし A が前に使ったものと同じでなければ前処理を施す
        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if (self.prev_A - A).nnz == 0:
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.bicg(
            coef_matrix,
            right_hand_side,
            rtol=0,
            atol=tolerance,
            M=diags(1 / coef_matrix.diagonal()),
            maxiter=100 * coef_matrix.shape[0],
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        # 疎行列にして保管しておく
        self.prev_coef_matrix_preprocessed = Csr(coef_matrix)
        return result


class BiCGStabLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """BIConjugate Gradient STABilized iteration で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """

    prev_coef_matrix_preprocessed: Csr = None
    method_name: str = "BiCGStab"

    def solve(self, A: np.ndarray, b: np.ndarray, tolerance: float = 10**-7, *args) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if (self.prev_A - A).nnz == 0:
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.bicgstab(
            coef_matrix,
            right_hand_side,
            rtol=0,
            atol=tolerance,
            M=diags(1 / coef_matrix.diagonal()),
            maxiter=100 * coef_matrix.shape[0],
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        self.prev_coef_matrix_preprocessed = Csr(coef_matrix)
        return result


class CGSLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """Conjugate Gradient Squared iteration で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """

    prev_coef_matrix_preprocessed: Csr = None
    method_name: str = "CGS"

    def solve(self, A: np.ndarray, b: np.ndarray, tolerance: float = 10**-7, *args) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if (self.prev_A - A).nnz == 0:
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.cgs(
            coef_matrix,
            right_hand_side,
            rtol=0,
            atol=tolerance,
            M=diags(1 / coef_matrix.diagonal()),
            maxiter=100 * coef_matrix.shape[0],
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        self.prev_coef_matrix_preprocessed = Csr(coef_matrix)
        return result


class QMRLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """Quasi-Minimal Residual iteration で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """

    prev_coef_matrix_preprocessed: Csr = None
    method_name: str = "Quasi-Minimal Residual method"

    def solve(self, A: np.ndarray, b: np.ndarray, tolerance: float = 10**-7, *args) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if (self.prev_A - A).nnz == 0:
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.qmr(
            coef_matrix,
            right_hand_side,
            rtol=0,
            atol=tolerance,
            # M1=diags(1 / coef_matrix.diagonal()),
            maxiter=100 * coef_matrix.shape[0],
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        self.prev_coef_matrix_preprocessed = Csr(coef_matrix)
        return result


class TFQMRLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """Transpose-Free Quasi-Minimal Residual iteration で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """

    prev_coef_matrix_preprocessed: Csr = None
    method_name: str = "TFQMR"

    def solve(self, A: np.ndarray, b: np.ndarray, tolerance: float = 10**-7, *args) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if (self.prev_A - A).nnz == 0:
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.tfqmr(
            coef_matrix,
            right_hand_side,
            rtol=0,
            atol=tolerance,
            # M=diags(1 / coef_matrix.diagonal()),
            maxiter=100 * coef_matrix.shape[0],
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        self.prev_coef_matrix_preprocessed = Csr(coef_matrix)
        return result
