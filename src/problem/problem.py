"""最適化問題に関する module"""

# 親クラスから子クラスへの参照があるモジュールなので, annotations は必要
from __future__ import annotations

import dataclasses

import numpy as np
import scipy
from scipy.sparse import csr_matrix as Csr
from scipy.sparse import hstack, vstack
from scipy.sparse import lil_matrix as Lil
from scipy.sparse.linalg import eigsh


class SettingProblemError(Exception):
    """問題を設定する際に例外が起こったら起こすエラー"""

    pass


@dataclasses.dataclass
class LinearProgrammingProblemStandard:
    """標準形の線形計画問題に関するクラス

    以下の問題について定数を格納する
    min c.T @ x
    s.t.
        A @ x = b
        x >= 0

    Args:
        name: 問題名. デフォルトは空白
    """

    A: Csr
    b: np.ndarray
    c: np.ndarray
    name: str = ""

    @property
    def n(self):
        """変数の次元数"""
        return self.A.shape[1]

    @property
    def m(self):
        """制約の数"""
        return self.A.shape[0]

    def __post_init__(self):
        """設定された定数の次元が正しいか確認し, 正しくなければエラーを返す"""
        m = self.m
        if (dim_b := self.b.shape[0]) != m:
            msg = f"制約の次元数が異なります. Aの行数 : {m}, bの次元 : {dim_b}"
            raise SettingProblemError(msg)
        n = self.n
        if (dim_c := self.c.shape[0]) != n:
            msg = f"目的関数の次元数が異なります. Aの列数 : {n}, cの次元 : {dim_c}"
            raise SettingProblemError(msg)

    def __eq__(self, other: object) -> bool:
        """要素が `np.array` なので, 標準の __eq__ メソッドだとエラーになる"""
        is_same_A = np.array_equal(self.A, other.A)
        is_same_b = np.array_equal(self.b, other.b)
        is_same_c = np.array_equal(self.c, other.c)
        return is_same_A and is_same_b and is_same_c

    @property
    def max_abs_A(self) -> float:
        """A の係数のうち最大の絶対値を出力"""
        return np.abs(self.A).max()

    @property
    def min_abs_A_nonzero(self) -> float:
        """A の係数のうち最小の絶対値(0は除く)を出力"""
        return np.abs(self.A[self.A != 0]).min()

    @property
    def condition_number_A(self) -> float:
        return np.linalg.cond(self.A.todense())

    @property
    def max_sqrt_eigen_value_AAT(self) -> float:
        max_eig_val = eigsh(self.A @ self.A.T, k=1, which="LM", return_eigenvectors=False)
        return np.sqrt(max_eig_val[0])

    @property
    def min_sqrt_eigen_value_AAT(self) -> float:
        """最小固有値は eigsh だと求めにくい（反復上限に達するとエラーを吐く）ので使わない方が吉"""
        min_eig_val = eigsh(self.A @ self.A.T, k=1, which="SM", return_eigenvectors=False)
        return np.sqrt(min_eig_val[0])

    def is_full_row_rank(self) -> bool:
        """制約行列 A が full row rank かを出力

        Returns:
            bool: A が full row rank であれば true
        """
        return np.linalg.matrix_rank(self.A.todense()) == self.m

    def objective_main(self, x: np.ndarray) -> float:
        """主問題の目的関数値の出力"""
        return self.c.T @ x

    def objective_dual(self, y: np.ndarray) -> float:
        """双対問題の目的関数値の出力"""
        return self.b.T @ y

    def residual_main_constraint(self, x: np.ndarray) -> np.ndarray:
        """主問題の制約に対する残渣ベクトルの出力"""
        return self.A @ x - self.b

    def residual_dual_constraint(self, y: np.ndarray, s: np.ndarray) -> np.ndarray:
        """双対問題の制約に対する残渣ベクトルの出力"""
        return self.A.T @ y + s - self.c

    # このメソッドは最適解自体を変えてしまうので削除
    # def create_A_row_normalized(self) -> LinearProgrammingProblemStandard:
    #     """A の各行のノルムを 1 に正規化した問題を出力

    #     Returns:
    #         LinearProgrammingProblemStandard: 正規化された問題
    #     """
    #     norm_each_row_A = np.linalg.norm(self.A, axis=1)
    #     return LinearProgrammingProblemStandard(self.A / norm_each_row_A[:, None], self.b / norm_each_row_A, self.c, self.name)

    def create_A_LU_factorized(self) -> tuple[LinearProgrammingProblemStandard, list[int]]:
        """A をLU分解した問題を出力
        Uの対角成分が0の場合, bの対応する行がすべて0であれば, その行をすべて削除して問題サイズを小さくする
        もし0でなければ実行不可能としてエラーを吐く

        Returns:
            LinearProgrammingProblemStandard: A=U, b=(PL)^(-1) とした問題.
                U の対角成分が0の行は抜いてある
            list[int]: もしUの対角成分が0だった場合にどの行が対象になったかを表す list
        """
        P, L, U = scipy.linalg.lu(self.A)
        b_factorized = np.linalg.inv(P) @ np.linalg.inv(L) @ self.b

        # U の対角成分が0の行を記録
        idx_zero_diag_element: list[int] = []
        for i in range(self.m):
            if np.all(np.isclose(U[i, :], 0)):
                if not np.isclose(b_factorized[i], 0):
                    raise SettingProblemError(
                        f"{i}行目において数値誤差を凌駕するほど実行不可能. U_i: {U[i, :]}, b_i: {b_factorized[i]}"
                    )
                idx_zero_diag_element.append(i)

        # 指定された行を省いて出力
        idx_leave_row = np.ones(self.m, dtype=bool)
        idx_leave_row[idx_zero_diag_element] = False
        A_new = U[idx_leave_row, :]
        b_new = b_factorized[idx_leave_row]

        return LinearProgrammingProblemStandard(A_new, b_new, self.c, self.name), idx_zero_diag_element


@dataclasses.dataclass
class LinearProgrammingProblem:
    """線形計画問題に関するクラス

    以下の問題について定数を格納する
    min c.T @ x
    s.t.
        A_E @ x = b_E
        A_G @ x >= b_G
        A_L @ x <= b_L
        x[LB_index] >= LB
        x[UB_index] <= UB

    Args:
        LB_index: 変数の下限が存在する場合のその添え字リスト.
            変数の下限が存在しない場合もあるため index を指定する必要がある
        LB: 変数の下限. LO_index 通りにソートされている
        UB_index: 変数の上限が存在する場合のその添え字リスト.
            変数の上限が存在しない場合もあるため index を指定する必要がある
        UB: 変数の上限. UP_index 通りにソートされている
        name: 問題名. デフォルトは空白
    """

    A_E: Lil
    b_E: np.ndarray
    A_G: Lil
    b_G: np.ndarray
    A_L: Lil
    b_L: np.ndarray
    LB_index: list[int]
    LB: np.ndarray
    UB_index: list[int]
    UB: np.ndarray
    c: np.ndarray
    name: str = ""

    @property
    def n(self):
        """変数の次元数

        A_* が空行列となる場合があるため, c の次元から取得
        """
        return self.c.shape[0]

    def __eq__(self, other: object) -> bool:
        """要素が `np.array` なので, 標準の __eq__ メソッドだとエラーになる

        TODO:
            * LB, UB でも判定するようにする
        """
        is_same_A_E = np.array_equal(self.A_E, other.A_E)
        is_same_b_E = np.array_equal(self.b_E, other.b_E)
        is_same_A_G = np.array_equal(self.A_G, other.A_G)
        is_same_b_G = np.array_equal(self.b_G, other.b_G)
        is_same_A_L = np.array_equal(self.A_L, other.A_L)
        is_same_b_L = np.array_equal(self.b_L, other.b_L)
        is_same_A = is_same_A_E and is_same_A_G and is_same_A_L
        is_same_b = is_same_b_E and is_same_b_G and is_same_b_L
        is_same_c = np.array_equal(self.c, other.c)
        return is_same_A and is_same_b and is_same_c

    # 以下、標準系に修正するための処理
    # データが入っていなくても実行可能なため classmethod
    @classmethod
    def reverse_non_lower_bound(
        cls, lb: np.ndarray, ub: np.ndarray, A: Lil, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Lil, np.ndarray]:
        """変数の下限が存在せず, 上限が存在する場合, 対応する変数の添え字の符号を反転させる

        Args:
            lb: 変数の下限. 下限がない次元は -inf
            ub: 変数の上限. 上限がない次元は inf
            A: A
            c: c

        Returns:
            np.ndarray: 変数の lower bound
            np.ndarray: 変数の upper bound
            Lil: A. のちにどの制約かによって区分けするが, ここでの出力はひとまとめにしたもの
            np.ndarray: c
        """
        # 下限が存在せず, 上限が存在する添え字の取得
        id_lb_inf = np.where(lb == -np.inf)[0]
        id_ub_non_inf = np.where(ub != np.inf)[0]
        indexes = list(set(id_lb_inf) & set(id_ub_non_inf))

        # 対象の添え字の符号を反転する
        lb_out = lb.copy()
        ub_out = ub.copy()
        lb_out[indexes], ub_out[indexes] = -ub[indexes], -lb[indexes]
        A_out = A.copy()
        A_out[:, indexes] = -A[:, indexes]
        c_out = c.copy()
        c_out[indexes] = -c[indexes]
        return lb_out, ub_out, A_out, c_out

    @classmethod
    def separate_free_variable(
        cls, lb: np.ndarray, ub: np.ndarray, A: Lil, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, Lil, np.ndarray]:
        """変数に下限も上限もない自由変数の場合, 新しい変数列を作成して正の部分と負の部分に分ける

        正部分に関する変数は同じ場所に残し, 負部分に関する変数は末尾に追加する
        正部分は, 下限を0に変更するのみ
        負部分は, 下限0, 上限inf, -A_{正部分}, c_{正部分} をそれぞれ追加する

        Args:
            lb: 変数の下限. 下限がない次元は -inf
            ub: 変数の上限. 上限がない次元は inf
            A: A
            c: c

        Returns:
            np.ndarray: 変数の lower bound
            np.ndarray: 変数の upper bound
            Lil: A
            np.ndarray: c
        """
        # 自由変数の添え字の取得
        id_lb_inf = np.where(lb == -np.inf)[0]
        id_ub_non_inf = np.where(ub == np.inf)[0]
        indexes = list(set(id_lb_inf) & set(id_ub_non_inf))
        # 出力を入力からコピー. このインスタンスに追加する
        lb_out = lb.copy()
        ub_out = ub.copy()
        A_out = A.copy()
        c_out = c.copy()
        # Aに列追加する際に reshape が必要なので, 行数を取っておく
        m = A.shape[0]

        # 各自由変数ごとに追加する
        for id_ in indexes:
            # 正部分の添え字
            lb_out[id_] = 0
            # 負部分の添え字（追加）
            lb_out = np.append(lb_out, 0)
            ub_out = np.append(ub_out, np.inf)
            A_out = hstack([A_out, -A[:, id_].reshape(m, 1)]).tolil()
            c_out = np.append(c_out, -c[id_])
        return lb_out, ub_out, A_out, c_out

    @classmethod
    def make_standard_A(cls, A_E: Lil, A_G: Lil, A_L: Lil, ub: np.ndarray) -> Csr:
        """等式制約, 不等式制約から標準形式の係数行列を作成する

        Args:
            A_E: 等式制約に関する係数行列
            A_G: 下限制約に関する係数行列
            A_L: 上限制約に関する係数行列
            ub: 変数の上限. 上限が存在しない次元は inf
        """
        lst_index_up = np.where(ub != np.inf)[0]

        # 各次元取得
        n = A_E.shape[1]
        m_e = A_E.shape[0]
        m_g = A_G.shape[0]
        m_l = A_L.shape[0]
        m_b = len(lst_index_up)

        # box constraint に関する単位行列を作成
        A_B = np.zeros([m_b, n])
        for i, index_up in enumerate(lst_index_up):
            A_B[i, index_up] = 1

        # 組み合わせて一つの行列に
        output = vstack(
            [
                hstack([A_E, np.zeros([m_e, m_g + m_l + m_b])]),
                hstack([A_G, -np.eye(m_g), np.zeros([m_g, m_l + m_b])]),
                hstack([A_L, np.zeros([m_l, m_g]), np.eye(m_l), np.zeros([m_l, m_b])]),
                np.concatenate([A_B, np.zeros([m_b, m_g + m_l]), np.eye(m_b)], 1),
            ]
        )
        return output.tocsr()

    @classmethod
    def make_standard_b(cls, A_E, A_G, A_L, b_E, b_G, b_L, lb, ub) -> np.ndarray:
        """等式制約, 不等式制約から標準形式の right hand side を作成する

        Args:
            A_E: 等式制約に関する係数行列
            A_G: 下限制約に関する係数行列
            A_L: 上限制約に関する係数行列
            b_E: 等式制約に関する右辺
            b_G: 下限制約に関する右辺
            b_L: 上限制約に関する右辺
            lb: 変数の下限値
            ub: 変数の上限値. inf であれば上限を設定する必要がない
        """
        lst_index_up = np.where(ub != np.inf)[0]

        b = np.concatenate([b_E - A_E @ lb, b_G - A_G @ lb, b_L - A_L @ lb, ub[lst_index_up] - lb[lst_index_up]])
        return b

    @classmethod
    def make_standard_c(cls, c: np.ndarray, m_GL: int, ub: np.ndarray) -> np.ndarray:
        """不等式制約から標準形式の目的関数係数を作成する

        Args:
            c: c
            m_GL: 上限, 下限不等式制約の数を合わせた数
            ub: 変数の上限. 上限が存在しない次元は inf
        """
        m_B = len(np.where(ub != np.inf)[0])
        return np.concatenate([c, np.zeros(m_GL + m_B)])

    def convert_standard(self) -> LinearProgrammingProblemStandard:
        """等式制約のみの標準形線形計画問題に修正する"""
        # 変数の正負, 自由変数において変更があるため A はまとめておく
        A: Lil = vstack([self.A_E, self.A_G, self.A_L]).tolil()

        # 変数の上下限が存在しない箇所は発散させておく
        lb = np.full(self.n, -np.inf)
        lb[self.LB_index] = self.LB
        ub = np.full(self.n, np.inf)
        ub[self.UB_index] = self.UB

        # 変数すべてに下限を設定
        lb_tmp, ub_tmp, A_tmp, c_tmp = self.separate_free_variable(*self.reverse_non_lower_bound(lb, ub, A, self.c))
        # 等式制約に統一するため不等式制約を別々にする
        m_e = self.A_E.shape[0]
        m_g = self.A_G.shape[0]
        m_l = self.A_L.shape[0]
        A_E = A_tmp[range(m_e), :]
        A_G = A_tmp[range(m_e, m_e + m_g), :]
        m_gl = m_g + m_l
        A_L = A_tmp[range(m_e + m_g, m_e + m_gl), :]

        # 変形して標準形式の A, b, c 作成
        A_out: Csr = self.make_standard_A(A_E, A_G, A_L, ub_tmp)
        b_out = self.make_standard_b(A_E, A_G, A_L, self.b_E, self.b_G, self.b_L, lb_tmp, ub_tmp)
        c_out = self.make_standard_c(c_tmp, m_gl, ub_tmp)

        return LinearProgrammingProblemStandard(A_out, b_out, c_out, self.name)
