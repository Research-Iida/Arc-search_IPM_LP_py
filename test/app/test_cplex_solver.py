import pytest

from src.app.cplex_solver import LPCPLEXSolver
from src.solver.optimization_parameters import OptimizationParameters
from src.utils.config_utils import test_section

from ..utils import make_test_LP_and_initial_point, problem_name


@pytest.mark.cplex
def test_run_cplex_solver():
    """CPLEX solver の実行テスト."""
    aLP, v_0 = make_test_LP_and_initial_point()
    parameters = OptimizationParameters.import_(test_section)
    solver = LPCPLEXSolver(test_section, parameters)

    sut = solver.run(aLP, v_0)

    assert sut.problem.name == problem_name
    assert sut.aSolvedSummary.is_solved
    # TODO: mu なども確認できるようにしたい
