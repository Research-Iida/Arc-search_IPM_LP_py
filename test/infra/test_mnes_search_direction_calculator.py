import pytest

from src.infra.julia.mnes_search_direction_calculator import JuliaMNESSearchDirectionCalculator
from src.infra.path_generator import PathGenerator
from src.infra.python.mnes_search_direction_calculator import MNESSearchDirectionCalculator
from src.infra.python.repository_problem import LPRepository
from src.solver.linear_system_solver.exact_linear_system_solver import ExactLinearSystemSolver
from src.utils.config_utils import test_section

path_generator = PathGenerator(test_section)


@pytest.mark.julia
@pytest.mark.skip("工事中")
def test_selecting_same_base_indexes():
    kb2_LP = LPRepository(path_generator).read_processed_LP("KB2")
    linear_system_solver = ExactLinearSystemSolver()

    sol_base_indexes = MNESSearchDirectionCalculator(linear_system_solver).select_base_indexes(kb2_LP)
    test_base_indexes = JuliaMNESSearchDirectionCalculator(linear_system_solver).select_base_indexes(kb2_LP)

    assert sol_base_indexes == test_base_indexes
