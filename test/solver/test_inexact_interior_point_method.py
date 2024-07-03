import pytest

from src.solver.inexact_interior_point_method import InexactArcSearchIPM, InexactLineSearchIPM
from src.solver.solved_checker import InexactSolvedChecker
from src.utils import config_utils

from .utils import make_test_LP_and_initial_point, problem_name

config_section = "TEST"
config_base = config_utils.read_config(section=config_section)
config_opt = config_utils.read_config(
    config_base.get("PATH_CONFIG") + config_base.get("CONFIG_OPTIMIZER"), section=config_section
)
test_tolerance = 10**-3
solved_checker = InexactSolvedChecker(test_tolerance, config_opt.getfloat("THRESHOLD_XS_NEGATIVE"), False)


def test_parameter_settings():
    test_instance = InexactArcSearchIPM(config_section=config_section)

    test_instance.solved_checker.stop_criteria_threshold == config_opt.get("STOP_CRITERIA_PARAMETER")


@pytest.mark.parametrize(
    "name, solver",
    [
        ("inexact arc CG", InexactArcSearchIPM(config_section, solved_checker)),
        ("inexact line CG", InexactLineSearchIPM(config_section, solved_checker)),
        ("exact arc", InexactArcSearchIPM("ARC_EXACT_NES", solved_checker)),
        ("inexact arc BiCG", InexactArcSearchIPM("INEXACT_ARC_BICG_NES", solved_checker)),
        ("inexact arc BiCGStab", InexactArcSearchIPM("INEXACT_ARC_BICGSTAB_NES", solved_checker)),
        ("inexact arc CGS", InexactArcSearchIPM("INEXACT_ARC_CGS_NES", solved_checker)),
        ("inexact arc QMR", InexactArcSearchIPM("INEXACT_ARC_QMR_NES", solved_checker)),
        ("inexact arc TFQMR", InexactArcSearchIPM("INEXACT_ARC_TFQMR_NES", solved_checker)),
    ],
)
def test_run(name, solver):
    """inexact IPM の実行テスト.
    収束条件が `test/utils.py` にある `solver_by_test_LP` とは異なるため,
    別で書き出す
    """
    aLP, v_0 = make_test_LP_and_initial_point()
    aSolvedDetail = solver.run(aLP, v_0)

    assert aSolvedDetail.problem.name == problem_name
    assert aSolvedDetail.aSolvedSummary.mu <= test_tolerance
    assert aSolvedDetail.aSolvedSummary.max_r_b <= test_tolerance
    assert aSolvedDetail.aSolvedSummary.max_r_c <= test_tolerance
