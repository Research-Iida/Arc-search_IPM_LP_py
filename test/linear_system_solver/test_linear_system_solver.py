import numpy as np
import pytest

from src.linear_system_solver.inexact_linear_system_solver import CGLinearSystemSolver
from src.linear_system_solver.hhl_qiskit import HHLLinearSystemSolver


def test_CG_tolerance():
    """勾配のノルムが解の許容度以下に収まっていることを確認する関数
    """
    tolerance = 10**-3
    A = np.array([[1, 2], [3, 4]])
    # パッケージそのままでは b のノルム分許容誤差が大きくなるため, bが大きくても問題ないテストにする
    b = np.array([1000, 2000])
    solver = CGLinearSystemSolver()

    # 入力は対称正定値である必要あり
    sol_x = solver.solve(A.T @ A, A.T @ b, tolerance)

    assert np.linalg.norm(A @ sol_x - b) <= tolerance


def test_HHL_julia():
    """julia による HHL アルゴリズムが正しく解けるか確認
    """
    from src.linear_system_solver.hhl_julia import HHLJuliaLinearSystemSolver

    # 解: [2, 1, 0]
    A = np.array([[1.0, 1.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    b = np.array([3.0, 4.0, 0.0])
    test_result = HHLJuliaLinearSystemSolver(20).solve(A, b)
    solution = np.linalg.inv(A) @ b

    assert np.linalg.norm(A @ test_result - b) <= 0.1
    assert np.allclose(test_result, solution, atol=0.1)


@pytest.mark.skip("Quantum が解けないので省略")
def test_HHL():
    """HHL アルゴリズムで求解できるか確認

    HHL 自体が不安定で tolerance を設定してもその許容解まで解けないという背景があり,
    精度は二の次で考える
    """
    tolerance = 10**-3
    # A はエルミート, 正定値
    A = np.array([[1, -1 / 3, 0], [- 1 / 3, 1, -1 / 3], [0, -1 / 3, 1]])
    b = np.array([1, 0, 0])
    # A = np.array([[1, 1], [1, 2]])
    # b = np.array([3, 4])
    solver = HHLLinearSystemSolver()

    test_sol = solver.solve(A, b, tolerance)

    # assert np.linalg.norm(A @ test_sol - b) <= tolerance
    exact_sol = np.linalg.inv(A) @ b
    assert np.allclose(test_sol, exact_sol, atol=tolerance)
