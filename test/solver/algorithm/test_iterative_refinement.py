from src.infra.algorithm_builder import AlgorithmBuilder
from src.solver.solver import LPSolver
from src.utils.config_utils import read_config, test_section

from .utils import make_test_LP_and_initial_point

config_base = read_config(section=test_section)
config_opt = read_config(config_base.get("PATH_CONFIG") + config_base.get("CONFIG_OPTIMIZER"), section=test_section)


def test_run__with_inexact_arc():
    """iterative refinement inexact arc-search の実行テスト.
    収束条件が `test/utils.py` にある `solver_by_test_LP` とは異なるため,
    別で書き出す
    """
    solver = LPSolver(AlgorithmBuilder(test_section).build("iterative_refinement"))
    aLP, v_0 = make_test_LP_and_initial_point()
    aSolvedDetail = solver.run(aLP, v_0)

    tolerance = config_opt.getfloat("STOP_CRITERIA_PARAMETER")
    assert aSolvedDetail.aSolvedSummary.mu <= tolerance
    assert aSolvedDetail.aSolvedSummary.max_r_b <= tolerance
    assert aSolvedDetail.aSolvedSummary.max_r_c <= tolerance
