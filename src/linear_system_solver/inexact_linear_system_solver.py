"""
探索方向における線形方程式を解くクラス.
Strategy パターン採用.

アルゴリズムごとに別の線形方程式ソルバーを使用できるようにクラス化
"""
from abc import ABCMeta

import numpy as np
from scipy.sparse import csc_matrix
import scipy.sparse.linalg as spla

from ..logger import get_main_logger
from .exact_linear_system_solver import AbstractLinearSystemSolver

logger = get_main_logger()


class AbstractInexactLinearSystemSolver(AbstractLinearSystemSolver, metaclass=ABCMeta):
    """線形方程式を inexact に解くクラスの super class
    """
    pass


class CGLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """共役勾配法(Conjugate Gradient)で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """
    prev_coef_matrix_preprocessed: csc_matrix = None
    method_name: str = "Conjugate Gradient method"

    def solve(
        self, A: np.ndarray, b: np.ndarray,
        tolerance: float = 10**-7, *args
    ) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        # 数値誤差により係数行列が対称にならない場合があるため, 少し修正
        coef_matrix = (A + A.T) / 2
        right_hand_side = b.copy()

        # もし A が前に使ったものと同じでなければ前処理を施す
        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if np.all(self.prev_A == A):
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.cg(
            coef_matrix, right_hand_side,
            rtol=0, atol=tolerance,
            M=np.diag(1 / np.diag(coef_matrix)),
            maxiter=100 * coef_matrix.shape[0]
        )
        logger.info(f"{self.method_name} end.")

        # 理論的には係数行列は半正定値になるはずだが, 誤差の範囲内におさまらない場合, 数値誤差の影響で半正定値でない可能性がある
        # その場合だけ, 最小固有値分摂動を行って再度解く（毎回固有値を求めるのはコストが大きいので回数は少なくしたい）
        if np.linalg.norm(coef_matrix @ result - right_hand_side) > tolerance:
            logger.warning(f"{self.method_name} cannot solve within the tolerance!")
            eigen_value, _ = np.linalg.eig(coef_matrix)
            if not np.all(eigen_value > 0):
                min_eig = min(eigen_value)
                logger.warning(f"Coefficient matrix of the LSS is not positive definite! min eigen value: {min_eig}")
                coef_perturbation = np.abs(min_eig) + 10**(-3)
                logger.info(f"Perturbation coefficient matrix by {coef_perturbation}.")
                coef_matrix += coef_perturbation * np.eye(coef_matrix.shape[0])

                result, info = spla.cg(
                    coef_matrix, right_hand_side,
                    tol=0, atol=tolerance,
                    M=np.diag(1 / np.diag(coef_matrix)),
                    maxiter=100 * coef_matrix.shape[0]
                )

        # cg法が解けなかった場合の warning
        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        # 疎行列にして保管しておく
        self.prev_coef_matrix_preprocessed = csc_matrix(coef_matrix)
        return result


class BiCGLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """双共役勾配法で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """
    prev_coef_matrix_preprocessed: csc_matrix = None
    method_name: str = "BiCG"

    def solve(
        self, A: np.ndarray, b: np.ndarray,
        tolerance: float = 10**-7, *args
    ) -> np.ndarray:
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
            if np.all(self.prev_A == A):
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.bicg(
            coef_matrix, right_hand_side,
            rtol=0, atol=tolerance,
            M=np.diag(1 / np.diag(coef_matrix)),
            maxiter=100 * coef_matrix.shape[0]
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        # 疎行列にして保管しておく
        self.prev_coef_matrix_preprocessed = csc_matrix(coef_matrix)
        return result


class BiCGStabLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """BIConjugate Gradient STABilized iteration で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """
    prev_coef_matrix_preprocessed: csc_matrix = None
    method_name: str = "BiCGStab"

    def solve(
        self, A: np.ndarray, b: np.ndarray,
        tolerance: float = 10**-7, *args
    ) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if np.all(self.prev_A == A):
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.bicgstab(
            coef_matrix, right_hand_side,
            rtol=0, atol=tolerance,
            M=np.diag(1 / np.diag(coef_matrix)),
            maxiter=100 * coef_matrix.shape[0]
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        self.prev_coef_matrix_preprocessed = csc_matrix(coef_matrix)
        return result


class CGSLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """Conjugate Gradient Squared iteration で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """
    prev_coef_matrix_preprocessed: csc_matrix = None
    method_name: str = "CGS"

    def solve(
        self, A: np.ndarray, b: np.ndarray,
        tolerance: float = 10**-7, *args
    ) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if np.all(self.prev_A == A):
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.cgs(
            coef_matrix, right_hand_side,
            rtol=0, atol=tolerance,
            M=np.diag(1 / np.diag(coef_matrix)),
            maxiter=100 * coef_matrix.shape[0]
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        self.prev_coef_matrix_preprocessed = csc_matrix(coef_matrix)
        return result


class QMRLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """Quasi-Minimal Residual iteration で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """
    prev_coef_matrix_preprocessed: csc_matrix = None
    method_name: str = "Quasi-Minimal Residual method"

    def solve(
        self, A: np.ndarray, b: np.ndarray,
        tolerance: float = 10**-7, *args
    ) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if np.all(self.prev_A == A):
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.qmr(
            coef_matrix, right_hand_side,
            rtol=0, atol=tolerance,
            # M1=np.diag(1 / np.diag(coef_matrix)),
            maxiter=100 * coef_matrix.shape[0]
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        self.prev_coef_matrix_preprocessed = csc_matrix(coef_matrix)
        return result


class TFQMRLinearSystemSolver(AbstractInexactLinearSystemSolver):
    """Transpose-Free Quasi-Minimal Residual iteration で線形方程式を解くクラス

    Attributes:
        prev_coef_matrix_preprocessed: `prev_A` に前処理を施した後の係数行列
    """
    prev_coef_matrix_preprocessed: csc_matrix = None
    method_name: str = "TFQMR"

    def solve(
        self, A: np.ndarray, b: np.ndarray,
        tolerance: float = 10**-7, *args
    ) -> np.ndarray:
        """線形方程式 Ax=b を共役勾配法によって解く

        Args:
            A (np.ndarray): 係数行列. CG法の入力となるので正定値対称行列である必要あり
            b (np.ndarray): 右辺のベクトル
            tolerance (float): 絶対誤差での許容度なので, ||Ax-b||<=tolerance となる
        """
        coef_matrix = A.copy()
        right_hand_side = b.copy()

        if self.prev_coef_matrix_preprocessed is not None and self.prev_A.shape == A.shape:
            if np.all(self.prev_A == A):
                logger.info("Use prev_A information.")
                coef_matrix = self.prev_coef_matrix_preprocessed.toarray()

        logger.info(f"{self.method_name} start.")
        result, info = spla.tfqmr(
            coef_matrix, right_hand_side,
            rtol=0, atol=tolerance,
            # M=np.diag(1 / np.diag(coef_matrix)),
            maxiter=100 * coef_matrix.shape[0]
        )
        logger.info(f"{self.method_name} end.")

        if info > 0:
            logger.warning(f"{self.method_name} cannot solve! # of iterations: {info}")
        elif info < 0:
            logger.warning("Illegal input or breakdown in Linear System.")

        self.prev_A = A.copy()
        self.prev_coef_matrix_preprocessed = csc_matrix(coef_matrix)
        return result
