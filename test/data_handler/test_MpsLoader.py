from pysmps import smps_loader as smps
import pytest

from src.utils import config_utils
from src.data_access import MpsLoader

config_ini = config_utils.read_config(section="TEST")
path_netlib = config_ini.get("PATH_NETLIB")


@pytest.fixture
def aMpsLoader():
    return MpsLoader(path_netlib)


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


def test_get_problem_names(aMpsLoader):
    """出力される SIF ファイル名はパスを含んでいないか, `.SIF` で終わっていないか"""
    for file in aMpsLoader.get_problem_names():
        assert not file.startswith(f"{path_netlib}")
        assert not file.endswith(".SIF")


@pytest.mark.slow("時間かかるので普段は skip")
def test_run_25FV47(aMpsLoader):
    """MpsLoader.run によりLPのインスタンスが作成されるか"""
    aLP = aMpsLoader.run("25FV47.SIF")
    assert aLP.A_E.shape[0] == aLP.b_E.shape[0]
    assert aLP.A_G.shape[0] == aLP.b_G.shape[0]
    assert aLP.A_L.shape[0] == aLP.b_L.shape[0]
    assert aLP.A_E.shape[1] == aLP.c.shape[0]
    assert aLP.A_G.shape[1] == aLP.c.shape[0]
    assert aLP.A_L.shape[1] == aLP.c.shape[0]
    aLPS = aLP.convert_standard()
    assert aLPS.A.shape[0] == aLPS.b.shape[0]
    assert aLPS.A.shape[1] == aLPS.c.shape[0]
