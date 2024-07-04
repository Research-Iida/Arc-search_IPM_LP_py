from src.solver.algorithm.iterative_refinement import IterativeRefinementMethod
from src.utils import config_utils

from .utils import make_test_LP_and_initial_point

config_section = "TEST"
config_base = config_utils.read_config(section=config_section)
config_opt = config_utils.read_config(
    config_base.get("PATH_CONFIG") + config_base.get("CONFIG_OPTIMIZER"), section=config_section
)


def test_run__with_inexact_arc():
    """iterative refinement inexact arc-search の実行テスト.
    収束条件が `test/utils.py` にある `solver_by_test_LP` とは異なるため,
    別で書き出す
    """
    solver = IterativeRefinementMethod(config_section)
    aLP, v_0 = make_test_LP_and_initial_point()
    aSolvedDetail = solver.run(aLP, v_0)

    tolerance = config_opt.getfloat("STOP_CRITERIA_PARAMETER")
    assert aSolvedDetail.aSolvedSummary.mu <= tolerance
    assert aSolvedDetail.aSolvedSummary.max_r_b <= tolerance
    assert aSolvedDetail.aSolvedSummary.max_r_c <= tolerance
