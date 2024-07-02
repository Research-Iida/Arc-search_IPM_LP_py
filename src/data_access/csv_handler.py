import os
from typing import Optional

import numpy as np
from dataclass_csv import DataclassWriter

from ..logger import get_main_logger
from ..problem import LinearProgrammingProblemStandard as LPS
from ..solver.solver import SolvedSummary
from ..utils import config_utils, file_util, str_util

logger = get_main_logger()


class CannotReadError(Exception):
    """この module で読み込みができない場合に発生するエラー"""

    pass


class CsvHandler:
    """csv ファイルの読み込み・書き込みに関する class"""

    def __init__(self, config_section: str = config_utils.default_section):
        """初期化. `data` ディレクトリへのパスを設定する"""
        config_ini = config_utils.read_config(section=config_section)

        self._path_data = config_ini.get("PATH_DATA")
        self._path_processed = config_ini.get("PATH_PROCESSED")
        self._path_result = config_ini.get("PATH_RESULT")

    def can_read_LP(self, problem_name: str) -> bool:
        """指定した問題がディレクトリに存在し, 読み取ることが可能か"""
        file_prefix = f"{self._path_processed}{problem_name}"

        can_read_A = os.path.exists(str_util.add_suffix_csv(file_prefix + "_A"))
        can_read_b = os.path.exists(str_util.add_suffix_csv(file_prefix + "_b"))
        can_read_c = os.path.exists(str_util.add_suffix_csv(file_prefix + "_c"))
        return can_read_A and can_read_b and can_read_c

    def read_LP(self, problem_name: str) -> LPS:
        """線形計画問題に関するcsvファイルを読み込み, 問題のクラスインスタンスを出力

        csv上で欠損している箇所は0を代入する
        """
        # 読み込めない場合, エラーを返す
        if not self.can_read_LP(problem_name):
            raise CannotReadError(f"{self._path_processed} 以下に {problem_name} が存在しません.")

        file_prefix = f"{self._path_processed}{problem_name}"

        def read(filename: str) -> np.ndarray:
            """各定数を読み込む処理"""
            return np.genfromtxt(filename, delimiter=",", filling_values=0)

        A = read(str_util.add_suffix_csv(file_prefix + "_A"))
        b = read(str_util.add_suffix_csv(file_prefix + "_b"))
        c = read(str_util.add_suffix_csv(file_prefix + "_c"))
        return LPS(A, b, c, problem_name)

    def write_numpy(self, filename: str, data: np.ndarray, path: Optional[str] = None):
        """numpy のデータをcsvファイルに書き出す

        Args:
            filename: ファイル名. `.csv` がなくともメソッドの中でつけるので問題ない
            data: 書き込み対象の numpy データ
            path: 書き込み先のpath. 指定がなければ `self._path_data` 直下に置く
        """
        # 書き込み先の決定
        if path is None:
            path = self._path_data

        fullpath_filename = str_util.add_suffix_csv(f"{path}{filename}")
        np.savetxt(fullpath_filename, data, delimiter=",")
        logger.info(f"{fullpath_filename} is written.")

    def write_LP(self, aLP: LPS, problem_name: str):
        """線形計画問題を csvファイルに書き出す

        前処理したものを書き出す前提のため, `processed` ディレクトリに書き出す

        TODO:
            * 0は書き下すとファイルサイズが大きくなるので, 欠損させるようにしたい
        """
        self.write_numpy(problem_name + "_A", aLP.A, self._path_processed)
        self.write_numpy(problem_name + "_b", aLP.b, self._path_processed)
        self.write_numpy(problem_name + "_c", aLP.c, self._path_processed)

    def write_SolvedSummary(
        self, lst_solved_summary: list[SolvedSummary], name: str, path: str, is_append: bool = False
    ):
        """最適化の実行によって得られた諸データを書き込む

        DataclassWriter を使用するため, self.write メソッドは使用しない

        Args:
            path: 書き込み先のディレクトリ
            is_append: 追記するか否か. 追記する場合は headerを抜く
        """
        file_util.create_dir_if_not_exists(path)
        filename = str_util.add_suffix_csv(f"{path}{name}")

        if is_append:
            mode = "a+"
        else:
            mode = "w"

        with open(filename, mode, newline="") as f:
            w = DataclassWriter(f, lst_solved_summary, SolvedSummary)
            w.write(skip_header=is_append)
