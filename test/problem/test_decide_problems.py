from src.infra.path_generator import PathGenerator
from src.infra.python.repository_problem import LPRepository
from src.problem.decide_solve_problem import decide_solved_problems
from src.utils.config_utils import test_section


def test_decide_solved_problems():
    num_problem = 1
    path_generator = PathGenerator(test_section)
    aLPRepository = LPRepository(path_generator)
    test_lst = decide_solved_problems(aLPRepository, num_problem)

    assert len(test_lst) == num_problem
    assert "OSA-30" not in test_lst
