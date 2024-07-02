"""Preprocessor class test"""
import numpy as np
import pytest

from src.problem.preprocessor import LPPreprocessor, ProblemInfeasibleError, ProblemUnboundedError


@pytest.fixture
def aPreprocessor():
    return LPPreprocessor()


def test_remove_empty_row(aPreprocessor):
    """Aの空行は, bが0であれば削除され,
    0でなければ実行不可能エラーが出されることを確認
    """
    test_sol = [1, 0]
    A = np.array([test_sol, [0, 0]])
    b = np.array([2, 0])
    test_A, test_b = aPreprocessor.remove_empty_row(A, b)
    np.testing.assert_array_equal(test_A, [test_sol])
    assert len(test_b) == 1

    b_error = np.array([1, 1])
    with pytest.raises(ProblemInfeasibleError):
        aPreprocessor.remove_empty_row(A, b_error)


def test_remove_duplicated_row(aPreprocessor):
    """重複が起きているAの行は削除され, 重複が起きているのにbの値がその係数倍と
    一致しない場合実行不可能エラーが出されることを確認
    """
    scalar_sol = 2
    test_sol = [1, 2]
    A = np.array([test_sol, [i * scalar_sol for i in test_sol]])
    base_b = 3
    b = np.array([base_b, base_b * scalar_sol])
    test_A, test_b = aPreprocessor.remove_duplicated_row(A, b)
    np.testing.assert_array_equal(test_A, [test_sol])
    assert len(test_b) == 1

    b_error = np.array([base_b, base_b + 1])
    with pytest.raises(ProblemInfeasibleError):
        aPreprocessor.remove_duplicated_row(A, b_error)


def test_remove_empty_column(aPreprocessor):
    """Aの空列は, cが0以上であれば削除され,
    負であれば実行不可能エラーが出されることを確認
    """
    test_sol = [1, 0]
    A = np.array([test_sol, [0, 0]]).T
    c = np.array([2, 0])
    test_A, test_c = aPreprocessor.remove_empty_column(A, c)
    # Aは2行1列の行列であることが正しい
    np.testing.assert_array_equal(test_A, np.array([test_sol]).T)
    assert len(test_c) == 1

    c_error = np.array([1, -1])
    with pytest.raises(ProblemUnboundedError):
        aPreprocessor.remove_empty_column(A, c_error)


def test_remove_duplicated_column(aPreprocessor):
    """まったく同じAの列は削除されることを確認"""
    test_sol = [1, 2]
    A = np.array([test_sol, test_sol]).T
    c = np.array([1, 2])
    test_A, test_c = aPreprocessor.remove_duplicated_column(A, c)
    np.testing.assert_array_equal(test_A, np.array([test_sol]).T)
    assert len(test_c) == 1


def test_rows_only_one_nonzero(aPreprocessor):
    """Aの第一行目が0以外の係数が1つしかない場合にindex 0 を出力できるか"""
    A = np.array([[2, 0, 0], [1, 3, 1]])
    test_lst = aPreprocessor.rows_only_one_nonzero(A)
    assert test_lst == [0]


def test_only_one_nonzero_elements_and_columns(aPreprocessor):
    """Aの第一行目が0以外の係数が1つしかない場合にindex 0 を出力できるか"""
    A = np.array([[2, 0, 0], [1, 3, 1]])
    lst_id = aPreprocessor.rows_only_one_nonzero(A)
    vec, ids = aPreprocessor.only_one_nonzero_elements_and_columns(A, lst_id)
    np.testing.assert_array_equal(vec, [2])
    assert ids == [0]


def test_remove_row_singleton(aPreprocessor):
    """Aの行の要素が1つしかないときは削除されることを確認"""
    test_sol = [3, 1]
    A = np.array([[2, 0, 0], [1] + test_sol])
    b = np.array([4, 3])
    c = np.ones(A.shape[1])
    test_A, test_b, test_c = aPreprocessor.remove_row_singleton(A, b, c)
    np.testing.assert_array_equal(test_A, [test_sol])
    assert test_b == 3 - 2
    np.testing.assert_array_equal(test_c, np.ones(test_A.shape[1]))

    # b / A が負になると実行不可能
    b = np.array([-1, 2])
    with pytest.raises(ProblemInfeasibleError):
        aPreprocessor.remove_row_singleton(A, b, c)


def test_find_zero_columns_in_A(aPreprocessor):
    """Aの2列目と3列目が足して0だった場合, リストで `[2]` （3列目）が帰ってくるか
    Aの1列目と3列目が足して0だった場合, リストで `[2]` （3列目）が帰ってくるか
    """
    A = np.array([[1, 1, -1], [2, -1, 1]])
    test_columns = aPreprocessor.find_zero_columns_in_A(A, 1)
    assert test_columns == [2]
    A = np.array([[1, 1, -1], [-1, 2, 1]])
    test_columns = aPreprocessor.find_zero_columns_in_A(A, 0)
    assert test_columns == [2]


def test_remove_free_variables(aPreprocessor):
    """Aの列が足して0になり, かつcの対応する添え字も同様であれば,
    制約は二つの変数を固定させるために減り, 変数は2つ減る
    """
    A = np.array([[1, 1, -1], [2, -1, 1]])
    b = np.array([2, 1])
    c = np.array([1, 3, -3])
    test_A, test_b, test_c = aPreprocessor.remove_free_variables(A, b, c)
    np.testing.assert_array_equal(test_A, [[2 - (-1) * 1 / 1]])
    assert test_b == 2 - (-1) * 1 / 1
    np.testing.assert_array_equal(test_c, [1 - 3 * 1 / 1])


def test_fix_variables_by_single_row(aPreprocessor):
    """bが0で制約の係数が正のみ, もしくは負のみの時に削除されることを確認"""
    test_sol = [3]
    A = np.array([[2, 1, 0], [1, 3] + test_sol])
    b = np.array([0, 3])
    c = np.ones(A.shape[1])
    test_A, test_b, test_c = aPreprocessor.fix_variables_by_single_row(A, b, c)
    np.testing.assert_array_equal(test_A, [test_sol])
    assert test_b == 3
    np.testing.assert_array_equal(test_c, np.ones(test_A.shape[1]))

    # bが負の時に制約が正の係数しか持たない場合実行不可能
    b = np.array([-1, 3])
    with pytest.raises(ProblemInfeasibleError):
        aPreprocessor.fix_variables_by_single_row(A, b, c)

    # bが正の時に制約が負の係数しか持たない場合実行不可能
    A = np.array([[-2, 1, 1], [-1, 0, -3]])
    with pytest.raises(ProblemInfeasibleError):
        aPreprocessor.fix_variables_by_single_row(A, b, c)


def test_fix_variables_by_multiple_rows(aPreprocessor):
    """bの値が同じ, もしくは足して0になる制約において, 制約の係数同士を引いた値が負もしくは正になる場合は
    その変数は0になることを確認
    """
    A = np.array([[2, 1], [2, 2], [1, 3]])
    b = np.array([1, 1, 3])
    c = np.ones(A.shape[1])
    test_A, test_b, test_c = aPreprocessor.fix_variables_by_multiple_rows(A, b, c)
    np.testing.assert_array_equal(test_A, [[2], [1]])
    np.testing.assert_array_equal(test_b, [1, 3])
    np.testing.assert_array_equal(test_c, np.ones(test_A.shape[1]))

    A = np.array([[2, 1], [-2, -2], [1, 3]])
    b = np.array([1, -1, 3])
    c = np.ones(A.shape[1])
    test_A, test_b, test_c = aPreprocessor.fix_variables_by_multiple_rows(A, b, c)
    np.testing.assert_array_equal(test_A, [[2], [1]])
    np.testing.assert_array_equal(test_b, [1, 3])
    np.testing.assert_array_equal(test_c, np.ones(test_A.shape[1]))


def test_fix_positive_variable_by_signs(aPreprocessor):
    A = np.array([[1, -1, -2], [3, 1, -1], [0, 2, -1]])
    b = np.array([1, 4, 0])
    c = np.ones(A.shape[1])
    test_A, test_b, test_c = aPreprocessor.fix_positive_variable_by_signs(A, b, c)
    np.testing.assert_array_equal(test_A, [[1 - 3 * (-1) / 1, -1 - 3 * (-2) / 1], [2, -1]])
    np.testing.assert_array_equal(test_b, [4 - 3 * 1 / 1, 0])
    np.testing.assert_array_equal(test_c, [1 - 1 * (-1) / 1, 1 - 1 * (-2) / 1])


def test_fix_singleton_by_two_rows(aPreprocessor):
    """2つの行の差が1列しか非0要素を持たない場合, その計算結果が正であれば
    変数を固定し, 負であれば実行不可能になることを確認
    """
    test_sol = [2, 1]
    A = np.array([[3] + test_sol, [2] + test_sol])
    b = np.array([4, 3])
    c = np.array(range(3))
    test_A, test_b, test_c = aPreprocessor.fix_singleton_by_two_rows(A, b, c)
    np.testing.assert_array_equal(test_A, [test_sol])
    assert test_b == 1
    np.testing.assert_array_equal(test_c, [1, 2])

    b = np.array([1, 2])
    with pytest.raises(ProblemInfeasibleError):
        aPreprocessor.fix_singleton_by_two_rows(A, b, c)
