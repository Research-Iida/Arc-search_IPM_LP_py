import glob
import os

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.infra.path_generator import PathGenerator
from src.infra.python.repository_problem import LPRepository
from src.problem import LinearProgrammingProblemStandard as LPS
from src.problem.repository import CannotReadError
from src.utils.config_utils import test_section

path_generator = PathGenerator(test_section)
path_netlib = path_generator.generate_path_netlib()

# ファイル名の設定
filename_written = "written"


@pytest.fixture
def aLPRepository() -> LPRepository:
    return LPRepository(test_section)


def test_get_problem_names(aLPRepository):
    """出力される SIF ファイル名はパスを含んでいないか, `.SIF` で終わっていないか"""
    for file in aLPRepository.get_problem_names():
        assert not file.startswith(f"{path_netlib}")
        assert not file.endswith(".SIF")


def test_run_25FV47(aLPRepository):
    """LPRepository によりLPのインスタンスが作成されるか"""
    aLP = aLPRepository.read_raw_LP("25FV47.SIF")
    assert aLP.A_E.shape[0] == aLP.b_E.shape[0]
    assert aLP.A_G.shape[0] == aLP.b_G.shape[0]
    assert aLP.A_L.shape[0] == aLP.b_L.shape[0]
    assert aLP.A_E.shape[1] == aLP.c.shape[0]
    assert aLP.A_G.shape[1] == aLP.c.shape[0]
    assert aLP.A_L.shape[1] == aLP.c.shape[0]
    aLPS = aLP.convert_standard()
    assert aLPS.A.shape[0] == aLPS.b.shape[0]
    assert aLPS.A.shape[1] == aLPS.c.shape[0]


def test_cannot_read_LP(aLPRepository):
    """ファイルが存在しない場合にエラーを返すか"""
    with pytest.raises(CannotReadError):
        aLPRepository.read_processed_LP("not_exist_file")


@pytest.fixture(scope="module")
def remove_written_file():
    """書き込みのテスト前にファイルが存在するとテストが通ったのかいなかわからないので,
    書き込む前に存在するファイルは削除する"""
    file_list = glob.glob(f"{path_generator.generate_path_data()}*/{filename_written}*")
    for filename in file_list:
        os.remove(filename)

    yield


def test_write_LP(aLPRepository, remove_written_file):
    """書き込んだファイルが全く同じオブジェクトを返すか"""
    sol_LP = LPS(A=csr_matrix([[1, 1], [0, 1]]), b=np.array([0, 1]), c=np.array([2, 0]))
    aLPRepository.write_LP(sol_LP, filename_written)
    test_LP = aLPRepository.read_processed_LP(filename_written)
    assert len((test_LP.A != sol_LP.A).data) == 0
    np.testing.assert_array_equal(test_LP.b, sol_LP.b)
    np.testing.assert_array_equal(test_LP.c, sol_LP.c)
    assert test_LP.name == filename_written


@pytest.mark.julia
def test_same_LP_between_pure_python_and_julia(aLPRepository):
    """pure python での実装と julia を使った実装が結果同じになることを確認
    KEN-07 を使用
    """
    from src.infra.julia.repository_problem import JuliaLPRepository

    problem_name = "KEN-07"
    sol_LP = aLPRepository.read_raw_LP(problem_name)
    test_LP = JuliaLPRepository(test_section).read_raw_LP(problem_name)
    assert test_LP == sol_LP
