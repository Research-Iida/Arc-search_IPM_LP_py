import numpy as np
import pytest
from scipy.sparse import csr_matrix as Csr

from src.app.algorithm_builder import AlgorithmBuilder
from src.problem import LinearProgrammingProblemStandard as LPS
from src.solver.variables import LPVariables
from src.utils.config_utils import test_section

from ...utils import solver_by_test_LP


@pytest.fixture
def algorithm():
    return AlgorithmBuilder(test_section).build("arc")


def test_mu():
    """μの計算の確認"""
    x = np.array([1, 2])
    s = np.array([3, 4])
    v = LPVariables(x, [], s)
    assert v.mu == (1 * 3 + 2 * 4) / 2


def test_residual_constraint():
    """制約の残渣の出力を確認"""
    A = np.eye(2)
    b = np.array([0, 1])
    c = np.array([1, 0])
    problem = LPS(Csr(A), b, c)

    x = np.ones(2)
    test_vector = problem.residual_main_constraint(x)
    np.testing.assert_array_equal(test_vector, [1, 0])

    y = np.zeros(2)
    s = np.ones(2)
    test_vector = problem.residual_dual_constraint(y, s)
    np.testing.assert_array_equal(test_vector, [0, 1])


def test_calc_first_derivatives(algorithm):
    """一次微分の出力を確認"""
    problem = LPS(Csr(np.eye(2)), np.array([0, 1]), np.array([1, 0]))
    v = LPVariables(np.ones(2), np.zeros(2), np.ones(2))
    x_dot, y_dot, s_dot = algorithm.calc_first_derivatives(v, problem)
    np.testing.assert_array_equal(y_dot, [0, 0])
    np.testing.assert_array_equal(s_dot, [0, 1])
    np.testing.assert_array_equal(x_dot, [1, 0])


def test_centering_parameter(algorithm):
    """sigma が正しく出力されているか確認

    mu^a は計算すると3になる
    """
    v = LPVariables(np.ones(2), [], np.ones(2))
    x_dot = np.array([2, 1])
    s_dot = np.array([2, 1])

    sigma = algorithm.centering_parameter(v, x_dot, s_dot)
    assert sigma == (1 / 8) ** 3


def test_calc_second_derivative(algorithm):
    """二次微分の値が計算されているか確認

    Note:
        * 解は手計算で行った
        * 一次微分値, および mu, sigma は計算しやすいように適当な値を代入
    """
    problem = LPS(Csr(np.eye(2)), np.array([0, 1]), np.array([1, 0]))
    v = LPVariables(np.ones(2), np.zeros(2), np.ones(2))
    x_ddot, y_ddot, s_ddot = algorithm.calc_second_derivative(
        v, np.ones(2), np.ones(2), np.ones(2), problem, mu=1, sigma=8
    )
    np.testing.assert_array_equal(y_ddot, [-6, -6])
    np.testing.assert_array_equal(s_ddot, -problem.A.T.dot(y_ddot))
    np.testing.assert_array_equal(x_ddot, [0, 0])


def test_run_line():
    """line search で求解できるか確認"""
    solver_by_test_LP(AlgorithmBuilder(test_section).build("line"))


def test_run_arc(algorithm):
    """求解できるか確認

    x の出力で0に近いものは削除される
    """
    solver_by_test_LP(algorithm)


def test_run_arc_restarting():
    solver_by_test_LP(AlgorithmBuilder(test_section).build("arc_restarting"))


def test_run_arc_restarting_proven():
    solver_by_test_LP(AlgorithmBuilder(test_section).build("arc_restarting_proven"))


def test_run_arc_restarting_guranteeing_main_residual_decreasing():
    solver_by_test_LP(AlgorithmBuilder("GUARANTEE_MAIN_RESIDUAL_DECREASING").build("arc_restarting"))
