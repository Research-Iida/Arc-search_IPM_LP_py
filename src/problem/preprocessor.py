"""LPに関する前処理を行う module"""
from typing import Optional

import numpy as np
from tqdm import tqdm

from src.problem import LinearProgrammingProblemStandard as LPS
from src.logger import get_main_logger


logger = get_main_logger()


class ProblemInfeasibleError(Exception):
    """前処理により実行不可能な問題であることがわかった場合に出すエラー"""
    pass


class ProblemUnboundedError(Exception):
    """前処理により最適値が発散してしまうことがわかった場合に出すエラー"""
    pass


class LPPreprocessor:
    """LPに関する前処理クラス"""
    def _remove_rows_and_columns(
        self, A: np.ndarray, b: np.ndarray = None, c: np.ndarray = None,
        rows_remove: set[int] = set(), columns_remove: set[int] = set()
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """削除する行, 列を指定して, A, b, c から削除する

        Args:
            b: 制約の右辺. 削除する必要がなければ入力しない
            c: 目的関数の係数. 削除する必要がなければ入力しない
            rows_remove: 削除行. 入力されなければ空集合
            columns_remove: 削除列. 入力されなければ空集合
        """
        # 残す行, 列を指定
        rows_out = [i for i in range(A.shape[0]) if i not in rows_remove]
        columns_out = [i for i in range(A.shape[1]) if i not in columns_remove]

        # 行が削除された結果
        A_out = A[np.ix_(rows_out, columns_out)]
        if b is not None:
            b_out = b[rows_out]
        else:
            b_out = None
        if c is not None:
            c_out = c[columns_out]
        else:
            c_out = None

        return A_out, b_out, c_out

    def remove_empty_row(
        self, A: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """空行で制約が存在しないAの行は削除する

        もしAの係数がないのにbが0以外の場合, どうやってもその制約は満たせないのでエラー
        """
        rows_remove = set()
        # A が空行になっているindexに対して処理
        for i in np.where(np.all(A == 0, axis=1))[0]:
            # もしAが空行なのにbが0でなければ実行不可能
            if b[i] != 0:
                raise ProblemInfeasibleError
            rows_remove.add(i)

        A_out, b_out, _ = self._remove_rows_and_columns(
            A, b=b, rows_remove=rows_remove
        )
        return A_out, b_out

    def remove_duplicated_row(
        self, A: np.ndarray, b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """重複のある制約を削除

        A_i = k*A_j の時, b_i = k*b_j ならば, jの制約はなくても同じなので削除してよい
        b_i != k*b_j ならば実行不可能になる
        """
        m = A.shape[0]

        def scalar_times_vector(
            vec_a: np.ndarray, vec_b: np.ndarray
        ) -> Optional[float]:
            """vec_b が vec_a の定数倍であるならばその値を, そうでなければ None を出力"""
            # 0の位置が正しくなければ定数倍ではない
            idx_not_zero_a = np.where(vec_a != 0)[0]
            idx_not_zero_b = np.where(vec_b != 0)[0]
            if len(idx_not_zero_a) != len(idx_not_zero_b):
                return None
            if (idx_not_zero_a != idx_not_zero_b).any():
                return None

            # ベクトル全体で割り算を行い, すべて同じ値でなければ定数倍ではない
            scalar_vec = vec_a[idx_not_zero_a] / vec_b[idx_not_zero_b]
            output = scalar_vec[0]
            if np.all(scalar_vec != output):
                return None
            return output

        # 削除する行を保持
        rows_remove = set()
        for i in tqdm(range(m - 1)):
            # すでに削除対象になっている場合省略する
            if i in rows_remove:
                continue
            for j in range(i + 1, m):
                # すでに削除対象になっている場合省略する
                if j in rows_remove:
                    continue
                # 対象とする行が定数倍になっているか確認
                k = scalar_times_vector(A[j, :], A[i, :])
                if k is None:
                    continue
                # 対象とする行が定数倍の場合, bの値も同じ定数倍になっていなければ実行不可能
                if b[j] != b[i] * k:
                    raise ProblemInfeasibleError
                # 削除対象として追加し次の添え字へ
                rows_remove.add(j)

        # 削除対象の行を省いて出力
        A_out, b_out, _ = self._remove_rows_and_columns(
            A, b=b, rows_remove=rows_remove
        )
        return A_out, b_out

    def remove_empty_column(
        self, A: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """空列で制約が存在しない変数は自由な値を取れるので, 削除する

        自由な値がとれる場合, cが正ならば0が最適値
        cが負ならばいくらでも小さくできてしまうので発散
        """
        columns_remove = set()
        # Aが空列になっている index に対してのみ処理
        for i in np.where(np.all(A == 0, axis=0))[0]:
            # もしAが空列なのにcが負であれば unbounded
            if c[i] < 0:
                raise ProblemUnboundedError
            columns_remove.add(i)

        A_out, _, c_out = self._remove_rows_and_columns(
            A, c=c, columns_remove=columns_remove
        )
        return A_out, c_out

    def remove_duplicated_column(
        self, A: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """重複する列を削除する

        行の場合と異なり, 全く同じ列でなければ削除をしない
        """
        n = A.shape[0]

        # 削除する列を保持
        columns_remove = set()
        for i in range(n - 1):
            # すでに削除対象になっている場合省略する
            if i in columns_remove:
                continue
            for j in range(i + 1, n):
                # すでに削除対象になっている場合省略する
                if j in columns_remove:
                    continue
                if np.all(A[:, i] == A[:, j]):
                    columns_remove.add(j)

        # 削除対象の列を省いて出力
        A_out, _, c_out = self._remove_rows_and_columns(
            A, c=c, columns_remove=columns_remove
        )
        return A_out, c_out

    def rows_only_one_nonzero(self, A: np.ndarray) -> list[int]:
        """Aの行のうち1つしか0以外の係数が存在しない行のインデックス取得

        A が1行m列の制約になった場合エラーとなるので, その時は axis を変更する
        """
        if A.shape[1] == 1:
            output = np.where(np.count_nonzero(A, axis=0) == 1)
        else:
            output = np.where(np.count_nonzero(A, axis=1) == 1)
        return output[0]

    def only_one_nonzero_elements_and_columns(
        self, A: np.ndarray, row_indexs_only_one_nonzero: list[int]
    ) -> tuple[np.ndarray, list[int]]:
        """1つしか係数がない行の係数のベクトル形式と, 列のインデックスを取得"""
        A_only_one_nonzero = A[row_indexs_only_one_nonzero, :]
        indexes_tmp = np.nonzero(A_only_one_nonzero)
        return A_only_one_nonzero[indexes_tmp], indexes_tmp[1]

    def remove_row_singleton(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """1つしか係数がかかっていない行は1つの値に定めることで次元数削除"""
        # 1つしか係数がない行の特定
        rows_remove = self.rows_only_one_nonzero(A)

        # 1つしか係数がない行がないならばそのまま出力
        if len(rows_remove) == 0:
            return A, b, c

        # 1つしか係数がない行の係数とその列を取得
        elements, columns_remove = self.only_one_nonzero_elements_and_columns(
            A, rows_remove
        )

        # 変数の確定. もし負の値になってしまったら実行不可能
        x = b[rows_remove] / elements
        if len(np.where(x < 0)[0]):
            raise ProblemInfeasibleError

        # 行が削除された結果
        A_new, _, c_new = self._remove_rows_and_columns(
            A, c=c, rows_remove=rows_remove, columns_remove=columns_remove
        )
        # bのみ値が変わるため, 別で計算する
        rows_output = [i for i in range(A.shape[0]) if i not in rows_remove]
        b_new = b[rows_output] - A[np.ix_(rows_output, columns_remove)].dot(x)

        # 行を削除した結果, 再び要素が1つだけの行ができるかもしれないので再度実行
        A_out, b_out, c_out = self.remove_row_singleton(A_new, b_new, c_new)
        return A_out, b_out, c_out

    def find_zero_columns_in_A(
        self, A: np.ndarray, column: int
    ) -> list[int]:
        """Aのj列に対して足して0になる列のリストを取得

        j-1列目までは確認していると考えて, j+1列目以降と比較する

        Args:
            A: 制約の係数行列
            column: Aのどの列と他の列が同じか参照するか（j列目）
        """
        A_j_minus_A = A[:, column].reshape(A.shape[0], 1) + A[:, (column + 1):]
        index_0 = np.where(np.all(A_j_minus_A == 0, axis=0))[0]
        return (index_0 + column + 1).tolist()

    def remove_free_variables(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Aの列が足して0になり, かつcの対応する添え字も同様であれば,
        2つの変数の差は自由変数として扱うことができるので,
        値を固定して次元を削減する
        """
        n = A.shape[1]
        for i in range(n - 1):
            logger.debug(f"Reference column: {i}")
            # 足して0になるAの列を見つける
            lst_columns = self.find_zero_columns_in_A(A, i)
            if not lst_columns:
                continue
            # 削除対象となるのは, for文で参照している列と足して0になるAの列の1つ
            columns_remove = [i, lst_columns[0]]
            cols_calc = [j for j in range(n) if j not in columns_remove]
            # 参照している列において係数が0以外である行を見つける
            lst_rows_nonzero_column = np.where(A[:, i] != 0)[0].tolist()
            # 係数が0しかない場合, 自由変数の削除としては扱わない
            if not lst_rows_nonzero_column:
                continue
            # 係数が0以外である最初の行のインデックスとその係数を取得
            row_remove = lst_rows_nonzero_column[0]
            A_alpha_i = A[row_remove, i]
            # 係数が0以外である各行に対して, 自由変数の係数と割り算
            rows_calc = lst_rows_nonzero_column[1:]
            # 対応する各要素に対して更新
            c = c - c[i] * A[row_remove, :] / A_alpha_i
            b = b - A[:, i] * b[row_remove] / A_alpha_i
            # 下記のAの行列計算は以下のfor文を行っている
            # for k in range(n):
            #     if k in columns_remove:
            #         continue
            #     tmp = A[rows_calc, i] / A_alpha_i * A[row_remove, k]
            #     A[rows_calc, k] = A[rows_calc, k] - tmp
            comb = np.ix_(rows_calc, cols_calc)
            logger.debug(f"Remove row: {row_remove}")
            logger.debug(f"Remove columns: {columns_remove}")
            A_beta_i = A[rows_calc, i].reshape(len(rows_calc), 1)
            A_alpha_k = A[row_remove, cols_calc].reshape(1, len(cols_calc))
            A_beta_i_A_alpha_k = A_beta_i.dot(A_alpha_k)
            A[comb] = A[comb] - A_beta_i_A_alpha_k / A_alpha_i
            A_row_removed = np.delete(A, row_remove, axis=0)
            A_new = np.delete(A_row_removed, columns_remove, axis=1)
            b_new = np.delete(b, row_remove, axis=0)
            c_new = np.delete(c, columns_remove, axis=0)
            # 自由変数を削除することで新たな自由変数が出現するかもしれないので再帰
            return self.remove_free_variables(A_new, b_new, c_new)
        # 足して0になる列の組がなければそのまま返す
        else:
            return A, b, c

    def fix_variables_by_single_row(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """bが負の時に制約が正の係数しか持たない, もしくはbが正の時に制約が負の係数しか
        もたない場合実行不可能. bが0の時に制約が正 or 負どちらかの係数しか持たない場合,
        解はすべて0として制約と変数を削除する
        """
        # 実行不可能性の確認
        rows_b_negative = np.where(b < 0)[0]
        if np.any(np.all(A[rows_b_negative, :] >= 0, axis=1)):
            raise ProblemInfeasibleError
        rows_b_positive = np.where(b > 0)[0]
        if np.any(np.all(A[rows_b_positive, :] <= 0, axis=1)):
            raise ProblemInfeasibleError

        # 制約, 変数の削除
        rows_remove = set()
        columns_remove = set()
        rows_b_zero = np.where(b == 0)[0]
        for row in rows_b_zero:
            row_A = A[row, :]
            columns_A_nonzero = np.where(row_A != 0)[0]
            # 係数が正 or 負のどちらかしか持たない場合, 解はすべて0とする
            if np.all(row_A >= 0) or np.all(row_A <= 0):
                rows_remove.add(row)
                columns_remove = columns_remove | set(columns_A_nonzero)

        # もし何も削除されなかった場合, そのまま入力を返す
        if not (rows_remove or columns_remove):
            return A, b, c

        # 残す制約, 変数を抽出
        A_new, b_new, c_new = self._remove_rows_and_columns(
            A, b=b, c=c, rows_remove=rows_remove, columns_remove=columns_remove
        )

        # 削除した結果, 再び同じ条件の行列ができるかもしれないので再度実行
        A_out, b_out, c_out = self.fix_variables_by_single_row(
            A_new, b_new, c_new
        )
        return A_out, b_out, c_out

    def fix_variables_by_multiple_rows(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """bの値が同じ制約において, 制約の係数同士を引いた場合に負か正の値しかない場合,
        その値は0にならなければ実行不可能
        bの値が足して0になる場合も同様. その場合制約の係数同士を足した場合になる
        """
        # 削除する行, 列の保持
        rows_remove = set()
        columns_remove = set()

        for i, b_i in tqdm(enumerate(b)):
            if i in rows_remove:
                continue

            # b が同じだった場合
            b_i_minus_b = b_i - b[i + 1:]
            rows_zero = np.where(b_i_minus_b == 0)[0] + (i + 1)
            for j in set(rows_zero) - rows_remove:
                A_i_minus_A_j = A[i, :] - A[j, :]
                if max(A_i_minus_A_j) <= 0 or min(A_i_minus_A_j) >= 0:
                    rows_remove.add(j)
                    columns_non_zero = np.where(A_i_minus_A_j != 0)[0]
                    columns_remove = columns_remove | set(columns_non_zero)

            # b が足して0だった場合
            b_i_plus_b = b_i + b[i + 1:]
            rows_zero = np.where(b_i_plus_b == 0)[0] + (i + 1)
            for j in set(rows_zero) - rows_remove:
                A_i_plus_A_j = A[i, :] + A[j, :]
                if max(A_i_plus_A_j) <= 0 or min(A_i_plus_A_j) >= 0:
                    rows_remove.add(j)
                    columns_non_zero = np.where(A_i_plus_A_j != 0)[0]
                    columns_remove = columns_remove | set(columns_non_zero)

        # 残す制約, 変数を抽出
        A_new, b_new, c_new = self._remove_rows_and_columns(
            A, b=b, c=c, rows_remove=rows_remove, columns_remove=columns_remove
        )

        return A_new, b_new, c_new

    def fix_positive_variable_by_signs(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray,
        recursive_num: int = 0
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ある行がbと同じ符号の係数が1つしかなく, それ以外は反対の符号, もしくは0の場合
        bと同じ符号の変数は他の変数との和で固定する

        Args:
            recursive_num: 再帰した回数. 再帰しすぎるとメモリエラーになるため, 50回を限度とする
        """
        if recursive_num > 50:
            return A, b, c

        # bが0以外の場合について, 条件に合致するAの行を取得
        for row in np.where(b != 0)[0]:
            # Aの要素がbの符号と同じ添え字を取得
            indexs_A_alpha_with_b = np.where(
                A[row, :] * b[row] / abs(b[row]) > 0
            )[0]
            # 要素が1つのみの場合, indexを取得し走査終了
            if len(indexs_A_alpha_with_b) == 1:
                remove_row = row
                remove_col = indexs_A_alpha_with_b[0]
                break
        # 該当する条件の行がなかった場合は入力された行列を出力へ
        else:
            return A, b, c

        # 各行, 列に対して値を更新
        A_alpha = A[remove_row, :]
        A_alpha_i = A_alpha[remove_col]
        A_beta_i = A[:, remove_col].reshape(A.shape[0], 1)
        A_new, b_new, c_new = self._remove_rows_and_columns(
            A - A_beta_i * A_alpha.reshape(1, A.shape[1]) / A_alpha_i,
            b=b - A[:, remove_col] * b[remove_row] / A_alpha_i,
            c=c - c[remove_col] * A_alpha / A_alpha_i,
            rows_remove={remove_row}, columns_remove={remove_col}
        )
        # 不要なオブジェクトを削除して再帰
        del A_alpha, A_beta_i, A, b, c
        num = recursive_num + 1
        return self.fix_positive_variable_by_signs(A_new, b_new, c_new, num)

    def fix_singleton_by_two_rows(
        self, A: np.ndarray, b: np.ndarray, c: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """2つの行の差が1列しか非0要素を持たない場合,
        1つの変数を固定することができるので固定することで変数を削除する

        もし固定した結果が負の値になる場合は実行不可能
        """
        for i in range(A.shape[0] - 1):
            A_i_minus_A = A[i, :] - A[(i + 1):, :]
            # 各行で非0要素になった数を集計
            counts_nonzero_by_row = np.count_nonzero(A_i_minus_A != 0, axis=1)
            # もしなければ次の行へ
            if 1 not in counts_nonzero_by_row:
                continue
            # あれば, その行が何行目か取得して, 変数を固定する
            index = counts_nonzero_by_row.tolist().index(1)
            remove_row = index + i + 1
            remove_column = np.where(A_i_minus_A[index, :] != 0)[0]
            x_denominator = A[i, remove_column] - A[remove_row, remove_column]
            x = (b[i] - b[remove_row]) / x_denominator
            # 固定した値が負であれば実行不可能
            if x < 0:
                raise ProblemInfeasibleError
            # 値の更新
            b_updated = b - A[:, remove_column].flatten() * x
            A_new, b_new, c_new = self._remove_rows_and_columns(
                A, b=b_updated, c=c,
                rows_remove=[remove_row], columns_remove=[remove_column]
            )
            # 削除した結果, 再び同じ条件の行列ができるかもしれないので再度実行
            return self.fix_singleton_by_two_rows(A_new, b_new, c_new)
        # 該当する条件の行がなかった場合はそのまま返す
        else:
            return A, b, c

    def run(self, problem: LPS) -> LPS:
        """前処理を実行し, 新しいLPのインスタンスを作成する

        一度実行した後にA, b, cが変形していれば, 再び0の行が出るなどして前処理が必要になるかもしれない
        そのため, 繰り返し出力する

        Returns:
            LPS: 前処理後の線形計画問題
        """
        logger.info("Start preprocessing.")
        A = problem.A.copy()
        b = problem.b.copy()
        c = problem.c.copy()

        logger.info("1. Start removing empty rows.")
        A, b = self.remove_empty_row(A, b)
        # logger.info("2. Start removing duplicated rows.")
        # A, b = self.remove_duplicated_row(A, b)
        logger.info("3. Start removing empty columns.")
        A, c = self.remove_empty_column(A, c)
        # logger.info("4. Start removing duplicated columns.")
        # A, c = self.remove_duplicated_column(A, c)
        logger.info("5. Start removing row singletons.")
        A, b, c = self.remove_row_singleton(A, b, c)
        # logger.info("6. Start removing free variables.")
        # A, b, c = self.remove_free_variables(A, b, c)
        logger.info("7. Start fixing variables by single row.")
        A, b, c = self.fix_variables_by_single_row(A, b, c)
        # logger.info("8. Start fixing variables by multiple rows.")
        # A, b, c = self.fix_variables_by_multiple_rows(A, b, c)
        logger.info("9. Start fixing positive variable by sings.")
        A, b, c = self.fix_positive_variable_by_signs(A, b, c)
        # logger.info("10. Start fixing singleton by two rows.")
        # A, b, c = self.fix_singleton_by_two_rows(A, b, c)

        # A, b, c について変更されているか確認し, もしされていなければ出力
        logger.info("Checking changing...")
        # まずは次元の確認から. 一致していることがわかった後に要素の確認をしないとサイズ違いでエラーとなる
        if A.shape == problem.A.shape and b.shape == problem.b.shape and c.shape == problem.c.shape:
            if np.all(A == problem.A) and np.all(b == problem.b) and np.all(c == problem.c):
                logger.info("End preprocessing.")
                if not problem.is_full_row_rank():
                    logger.warning("This problem is not full row rank!")
                return problem
        # されていればもう一度実行
        logger.info("Restart Preprocessing for changing coefficients.")
        # 不要なオブジェクトはメモリを圧迫するので削除
        name = problem.name
        del problem
        return self.run(LPS(A, b, c, name))
