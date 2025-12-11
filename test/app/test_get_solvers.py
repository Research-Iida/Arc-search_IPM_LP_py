from pathlib import Path

import pytest

from src.app.get_solvers import SolverSelectionError, get_solvers, load_solver_info


def test_raise_error_when_solver_name_does_not_exist():
    target_algorithms = {"solver_a": ["test_section"]}

    with pytest.raises(SolverSelectionError) as e_sut:
        get_solvers(target_algorithms, "solver_b", None)

    assert "solver_b" in str(e_sut)


def test_load_solver_info():
    path_solver_info = Path(__file__).with_name("solver_info.json")

    sut = load_solver_info(path_solver_info)

    assert "solver_a" in sut
    assert "section_a" in sut["solver_a"]
