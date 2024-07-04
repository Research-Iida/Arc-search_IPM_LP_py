"""`tests` ディレクトリ内のファイルを実行する際によく使用するものをまとめたファイル"""

import math

import numpy as np
from scipy.sparse import csr_matrix as Csr

from src.problem import LinearProgrammingProblemStandard as LPS
from src.solver.optimization_parameters import OptimizationParameters
from src.solver.solver import LPSolver
from src.solver.variables import LPVariables

problem_name = "test"
config_section = "TEST"
parameter = OptimizationParameters.import_(config_section)


def make_test_LP_and_initial_point() -> tuple[LPS, LPVariables]:
    """テストスクリプト共通で使用する線形計画問題と初期点を作成

    最適解は x_1=4, x_2=2, y1=y2=-1/3, s_1=s_2=0
    最適値は -6
    初期点は x=s=10, y=0
    """
    A = Csr(np.array([[1.0, 2.0], [2.0, 1.0]]))
    b = np.array([8, 10])
    c = np.array([-1, -1])

    v_0 = LPVariables(np.ones(2) * 10, np.zeros(2), np.ones(2) * 10)
    return LPS(A, b, c, problem_name), v_0


def solver_by_test_LP(aLPSolver: LPSolver):
    """テスト用問題をソルバーに解かせ, 結果は最適解と同じか確認する

    テストの問題名も同じになっているか確認する
    """
    aLP, v_0 = make_test_LP_and_initial_point()
    aSolvedDetail = aLPSolver.run(aLP, v_0)

    # 停止条件の値より1個上の小数点まで誤差とみなす
    tolerance = parameter.STOP_CRITERIA_PARAMETER
    variables_tolerance_decimal = -int(np.log10(tolerance) + 1)
    np.testing.assert_almost_equal(aSolvedDetail.v.x, [4, 2], decimal=variables_tolerance_decimal)
    np.testing.assert_almost_equal(aSolvedDetail.v.y, [-1 / 3] * 2, decimal=variables_tolerance_decimal)
    np.testing.assert_almost_equal(aSolvedDetail.v.s, [0, 0], decimal=variables_tolerance_decimal)
    assert math.isclose(aSolvedDetail.aSolvedSummary.obj, -6, abs_tol=tolerance)
    assert aSolvedDetail.aSolvedSummary.problem_name == problem_name
