import glob
import os

import numpy as np
import pytest
from pysmps import smps_loader as smps
from scipy.sparse import csr_matrix

from src.problem import LinearProgrammingProblemStandard as LPS
from src.problem.repository import CannotReadError, LPRepository
from src.utils.config_utils import read_config, test_section

config_ini = read_config(section=test_section)
path_netlib = config_ini.get("PATH_NETLIB")
path_data = config_ini.get("PATH_DATA")

# ファイル名の設定
filename_written = "written"


@pytest.fixture
def aLPRepository() -> LPRepository:
    return LPRepository(test_section)


def test_load_mps_KB2():
    """`pysmps` モジュールの `load_mps` メソッドがどのような振る舞いをするか確認する"""
    test_obj = smps.load_mps(f"{path_netlib}KB2.SIF")

    # 制約数の確認
    m = len(test_obj[2])
    A = test_obj[7]
    assert m == A.shape[0]

    # 変数次元数の確認
    n = len(test_obj[3])
    c = test_obj[6]
    assert n == c.shape[0]
    assert n == A.shape[1]

    # 制約の種類の確認
    # Equal, Grater than, Lower than
    types_constraint = test_obj[5]
    for type_const in types_constraint:
        assert type_const in {"E", "G", "L"}

    # KB2 は制約が設定されていない. 0以上か0以下か, もしくは0か
    name_constraints = test_obj[8]
    assert len(name_constraints) == 0

    # 変数の上下限制約
    # 同じ変数に2つ以上異なる上下限は設定されないはず
    name_bounds = test_obj[10]
    assert len(name_bounds) <= 1
    # KB2 では下限が設定されていないので, すべて0が下限になる
    dct_bound = test_obj[11][name_bounds[0]]
    lst_lb = dct_bound["LO"]
    assert lst_lb.shape[0] == n
    for bound in lst_lb:
        assert bound == 0
    # 上限は設定されているものとされていないものがあり, 設定されているものは inf
    assert dct_bound["UP"].shape[0] == n


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
    file_list = glob.glob(f"{path_data}*/{filename_written}*")
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
