
import numpy as np
from scipy.linalg import toeplitz

from ..solver.variables import LPVariables
from ..problem.problem import LinearProgrammingProblemStandard as LPS


def generate_problem(n: int, m: int) -> tuple[LPS, LPVariables]:
    """変数サイズと制約数を与えて LP を作成

    Args:
        n (int): 変数サイズ
        m (int): 制約数

    Returns:
        LPS: 問題
        LPVariables: 最適解
    """
    # x,s においてどの index が0になるか決める
    mask = [1 if ind < m else 0 for ind in range(n)]
    np.random.shuffle(mask)

    # 最適解の決定
    opt_x = np.multiply(np.random.rand(n), mask)
    opt_s = np.random.rand(n)
    opt_s = opt_s - np.multiply(opt_s, mask)
    opt_y = np.random.rand(m) - 0.5

    # A は各行で log(m) のスパース性を持ち full row rank
    # A = np.random.rand(m, n) - 0.5
    n_nonzero = int(np.log2(m))
    nonzero_elements = np.random.rand(n_nonzero) - 0.5
    A = toeplitz(np.concatenate([[nonzero_elements[0]], np.zeros(m - 1)]), np.concatenate([nonzero_elements, np.zeros(n - n_nonzero)]))
    b = A @ opt_x
    c = A.T @ opt_y + opt_s

    return LPS(A, b, c, f"random_problem_n_{n}_m_{m}"), LPVariables(opt_x, opt_y, opt_s)
