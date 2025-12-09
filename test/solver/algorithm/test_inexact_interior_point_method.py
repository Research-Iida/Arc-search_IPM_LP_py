import pytest

from src.infra.algorithm_builder import AlgorithmBuilder
from src.solver.algorithm.algorithm import ILPSolvingAlgorithm
from src.solver.optimization_parameters import OptimizationParameters
from src.utils.config_utils import read_config, test_section

from .utils import make_test_LP_and_initial_point, problem_name


def test_parameter_settings():
    test_instance = AlgorithmBuilder(test_section).build("inexact_arc")
    config_base = read_config(section=test_section)
    config_opt = read_config(
        config_base.get("PATH_CONFIG") + config_base.get("CONFIG_OPTIMIZER"), section=test_section
    )

    test_instance.solved_checker.stop_criteria_threshold == config_opt.get("STOP_CRITERIA_PARAMETER")


@pytest.mark.parametrize(
    "name, algorithm",
    [
        ("inexact arc CG", AlgorithmBuilder(test_section).build("inexact_arc")),
        ("inexact line CG", AlgorithmBuilder(test_section).build("inexact_line")),
        ("exact arc", AlgorithmBuilder("ARC_EXACT_NES").build("inexact_arc")),
        ("inexact arc BiCG", AlgorithmBuilder("INEXACT_ARC_BICG_NES").build("inexact_arc")),
        ("inexact arc BiCGStab", AlgorithmBuilder("INEXACT_ARC_BICGSTAB_NES").build("inexact_arc")),
        ("inexact arc CGS", AlgorithmBuilder("INEXACT_ARC_CGS_NES").build("inexact_arc")),
        ("inexact arc QMR", AlgorithmBuilder("INEXACT_ARC_QMR_NES").build("inexact_arc")),
        ("inexact arc TFQMR", AlgorithmBuilder("INEXACT_ARC_TFQMR_NES").build("inexact_arc")),
        ("inexact arc CG without proof", AlgorithmBuilder(test_section).build("inexact_arc_without_proof")),
    ],
)
def test_run(name, algorithm: ILPSolvingAlgorithm):
    """inexact IPM の実行テスト.
    収束条件が `test/utils.py` にある `solver_by_test_LP` とは異なるため,
    別で書き出す
    """
    aLP, v_0 = make_test_LP_and_initial_point()
    parameters = OptimizationParameters.import_(test_section)
    test_tolerance = parameters.STOP_CRITERIA_PARAMETER

    aSolvedDetail = algorithm.run(aLP, v_0)

    assert aSolvedDetail.problem.name == problem_name
    assert aSolvedDetail.aSolvedSummary.mu <= test_tolerance
    assert aSolvedDetail.aSolvedSummary.max_r_b <= test_tolerance
    assert aSolvedDetail.aSolvedSummary.max_r_c <= test_tolerance
