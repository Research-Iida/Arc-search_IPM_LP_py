from src.solver.algorithm.interior_point_method_with_restarting_strategy import (
    ArcSearchIPMWithRestartingStrategy,
    ArcSearchIPMWithRestartingStrategyProven,
)

from .utils import solver_by_test_LP


def test_run():
    solver_by_test_LP(ArcSearchIPMWithRestartingStrategy("TEST"))


def test_run_proven():
    solver_by_test_LP(ArcSearchIPMWithRestartingStrategyProven("TEST"))


def test_run_guranteeing_main_residual_decreasing():
    solver_by_test_LP(ArcSearchIPMWithRestartingStrategy("GUARANTEE_MAIN_RESIDUAL_DECREASING"))
