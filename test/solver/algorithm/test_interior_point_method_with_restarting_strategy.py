from src.solver.algorithm.algorithm_builder import AlgorithmBuilder
from src.utils.config_utils import test_section

from .utils import solver_by_test_LP


def test_run():
    solver_by_test_LP(AlgorithmBuilder(test_section).build("arc_restarting"))


def test_run_proven():
    solver_by_test_LP(AlgorithmBuilder(test_section).build("arc_restarting_proven"))


def test_run_guranteeing_main_residual_decreasing():
    solver_by_test_LP(AlgorithmBuilder("GUARANTEE_MAIN_RESIDUAL_DECREASING").build("arc_restarting"))
