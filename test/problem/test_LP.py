"""LinearProgrammingProblem class test"""

import numpy as np
import pytest
from scipy.sparse import lil_matrix as Lil

from src.problem import LinearProgrammingProblem as LP
from src.problem import LinearProgrammingProblemStandard as LPS
from src.problem import SettingProblemError


def test_set_problem_error():
    """問題の次元数が異なる場合にエラーを起こせるか"""
    A = np.ones([4, 3])
    b = np.ones(2)
    c = np.ones(3)
    with pytest.raises(SettingProblemError):
        LPS(A, b, c)


def test_reverse_non_lower_bound():
    """変数の下限が存在せず, 上限は存在する場合, 符号を反転するか

    float 型でないと np.inf が働かなくなるので注意
    """
    lb = np.array([-np.inf, 1.0])
    ub = np.array([1.0, 2.0])
    A = np.array([[1, 2], [3, 4]])
    c = np.array([1, 1])
    test_lb, test_ub, test_A, test_c = LP.reverse_non_lower_bound(lb, ub, A, c)
    np.testing.assert_array_equal(test_lb, [-1, 1])
    np.testing.assert_array_equal(test_ub, [np.inf, 2])
    np.testing.assert_array_equal(test_A, [[-1, 2], [-3, 4]])
    np.testing.assert_array_equal(test_c, [-1, 1])


def test_separate_free_variable():
    """変数に下限も上限もない, 自由変数の場合, 新しい変数列が作成されるか"""
    lb = np.array([-np.inf, 1.0])
    ub = np.array([np.inf, 2.0])
    A = Lil(np.array([[1, 2], [3, 4]]))
    c = np.array([1, 1])
    test_lb, test_ub, test_A, test_c = LP.separate_free_variable(lb, ub, A, c)
    np.testing.assert_array_equal(test_lb, [0, 1, 0])
    np.testing.assert_array_equal(test_ub, [np.inf, 2, np.inf])
    np.testing.assert_array_equal(test_A.toarray(), [[1, 2, -1], [3, 4, -3]])
    np.testing.assert_array_equal(test_c, [1, 1, -1])


def test_make_A():
    """LP 形式から係数行列Aが出力されるか"""
    A_E = Lil([[1, 0], [1, 1]])
    A_G = Lil([[2, 3]])
    A_L = Lil([[4, 5], [6, 7], [8, 9]])
    ub = np.array([np.inf, 1])
    A = LP.make_standard_A(A_E, A_G, A_L, ub)

    m_g = A_G.shape[0]
    m_l = A_L.shape[0]
    m_b = len(np.where(ub != np.inf)[0])
    sol_A = np.array(
        [
            [1, 0] + [0] * (m_g + m_l + m_b),
            [1, 1] + [0] * (m_g + m_l + m_b),
            [2, 3] + [-1] + [0] * (m_l + m_b),
            [4, 5] + [0] + [1, 0, 0] + [0] * m_b,
            [6, 7] + [0] + [0, 1, 0] + [0] * m_b,
            [8, 9] + [0] + [0, 0, 1] + [0] * m_b,
            [0, 1] + [0] * (m_g + m_l) + [1],
        ]
    )
    np.testing.assert_almost_equal(A.toarray(), sol_A)


def test_make_b():
    """SIF file 形式から right hand side b が出力されるか"""
    A_E = np.array([[1, 0], [1, 1]])
    A_G = np.array([[2, 1]])
    A_L = np.array([[4, 5], [6, 7], [8, 9]])
    lb = np.array([0, 1])

    b_E = np.array([1, 2])
    b_G = np.array([3])
    b_L = np.array([4, 5, 6])
    ub = np.array([np.inf, 7])

    b = LP.make_standard_b(A_E, A_G, A_L, b_E, b_G, b_L, lb, ub)
    sol_b = np.array([1, 1, 2, -1, -2, -3, 6])
    np.testing.assert_almost_equal(b, sol_b)


def test_make_c():
    c = np.array([1, 2])
    m_GL = 1 + 3
    lst_index_up = np.array([1])

    test_c = LP.make_standard_c(c, m_GL, lst_index_up)
    sol_c = np.array([1, 2] + [0, 0, 0, 0, 0])
    np.testing.assert_almost_equal(test_c, sol_c)


def test_convert_standard():
    """通常の線形計画問題を標準形に直せるか"""
    A_E = Lil([[0, 1]])
    b_E = np.array([2])
    A_G = Lil([[2, 3]])
    b_G = np.array([4])
    A_L = Lil([]).reshape([0, 2])
    b_L = np.array([])
    LB_index = []
    LB = np.array([])
    UB_index = [0]
    UB = np.array([5])
    c = np.array([6, 7])
    aLP = LP(A_E, b_E, A_G, b_G, A_L, b_L, LB_index, LB, UB_index, UB, c)
    test_LP = aLP.convert_standard()
    np.testing.assert_array_equal(test_LP.A.todense(), [[0, 1, -1, 0], [-2, 3, -3, -1]])
    np.testing.assert_array_equal(test_LP.b, [2, 4 - (-2 * -5)])
    np.testing.assert_array_equal(test_LP.c, [-6, 7, -7, 0])
