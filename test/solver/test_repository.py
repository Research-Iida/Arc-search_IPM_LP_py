import os
import shutil
from pathlib import Path

import pytest

from src.infra.repository_solved_data import SolvedDataRepository
from src.solver.solver import SolvedSummary
from src.utils.config_utils import read_config, test_section

# ファイル名の設定
filename_read = "test_problem"
filename_written = "test_written"


@pytest.fixture
def aSolvedDataRepository() -> SolvedDataRepository:
    return SolvedDataRepository(test_section)


@pytest.fixture
def remove_written_directory() -> Path:
    config_ini = read_config(section=test_section)
    path_result = config_ini.get("PATH_RESULT")
    path_output = Path(f"{path_result}test_SolvedDataRepository/")
    shutil.rmtree(path_output) if os.path.exists(path_output) else None

    return path_output


def test_write_SolvedSummary(aSolvedDataRepository, remove_written_directory):
    """最適化の実行によって得られた諸データを書き込めるか"""
    aSolvedSummary = SolvedSummary(
        "test_problem",
        "test_solver",
        "TEST",
        False,
        3,
        2,
        True,
        30,
        False,
        0.14,
        False,
        0,
        0.001,
        0,
        0,
    )
    aSolvedDataRepository.write_SolvedSummary([], filename_written, path=remove_written_directory)
    aSolvedDataRepository.write_SolvedSummary(
        [aSolvedSummary], filename_written, path=remove_written_directory, is_append=True
    )
