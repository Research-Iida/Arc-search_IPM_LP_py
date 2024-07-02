import glob
import os
import shutil

import numpy as np
import pytest

from src.data_access import CsvHandler
from src.data_access.csv_handler import CannotReadError
from src.solver.solver import SolvedSummary
from src.utils import config_utils

config_section = "TEST"
config_ini = config_utils.read_config(section=config_section)
path_data = config_ini.get("PATH_DATA")
path_result = config_ini.get("PATH_RESULT")

# ファイル名の設定
filename_read = "test_problem"
filename_written = "test_written"


@pytest.fixture
def aCsvHandler() -> CsvHandler:
    return CsvHandler(config_section=config_section)


def test_read_LP(aCsvHandler):
    """LPを問題なく読み込むことは可能か"""
    test_LP = aCsvHandler.read_LP(filename_read)
    np.testing.assert_array_equal(test_LP.A, np.array([[1, 1], [0, 1]]))
    np.testing.assert_array_equal(test_LP.b, np.array([0, 1]))
    np.testing.assert_array_equal(test_LP.c, np.array([2, 0]))


def test_cannot_read_LP(aCsvHandler):
    """ファイルが存在しない場合にエラーを返すか"""
    with pytest.raises(CannotReadError):
        aCsvHandler.read_LP(filename_read + "_")


@pytest.fixture(scope="module")
def remove_written_file():
    """書き込みのテスト前にファイルが存在するとテストが通ったのかいなかわからないので,
    書き込む前に存在するファイルは削除する"""
    file_list = glob.glob(f"{path_data}*/{filename_written}*")
    for filename in file_list:
        os.remove(filename)

    yield


def test_write_LP(aCsvHandler, remove_written_file):
    """書き込んだファイルが全く同じオブジェクトを返すか"""
    sol_LP = aCsvHandler.read_LP(filename_read)
    aCsvHandler.write_LP(sol_LP, filename_written)
    test_LP = aCsvHandler.read_LP(filename_written)
    np.testing.assert_array_equal(test_LP.A, sol_LP.A)
    np.testing.assert_array_equal(test_LP.b, sol_LP.b)
    np.testing.assert_array_equal(test_LP.c, sol_LP.c)


@pytest.fixture
def remove_written_directory():
    path_output = f"{path_result}test_CsvHandler/"
    shutil.rmtree(path_output) if os.path.exists(path_output) else None

    yield path_output


def test_write_SolvedSummary(aCsvHandler, remove_written_directory):
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
    aCsvHandler.write_SolvedSummary([], filename_written, path=remove_written_directory)
    aCsvHandler.write_SolvedSummary([aSolvedSummary], filename_written, path=remove_written_directory, is_append=True)
