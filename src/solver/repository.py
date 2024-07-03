from pathlib import Path

import numpy as np
from dataclass_csv import DataclassWriter

from ..logger import get_main_logger
from ..utils import file_util, str_util
from .solved_data import SolvedDetail, SolvedSummary

logger = get_main_logger()


class SolvedDataRepository:
    def write_SolvedSummary(
        self, lst_solved_summary: list[SolvedSummary], name: str, path: Path, is_append: bool = False
    ):
        """最適化の実行によって得られた諸データを書き込む

        DataclassWriter を使用するため, self.write メソッドは使用しない

        Args:
            path: 書き込み先のディレクトリ
            is_append: 追記するか否か. 追記する場合は headerを抜く
        """
        file_util.create_dir_if_not_exists(path)
        filename = str_util.add_suffix_csv(name)

        if is_append:
            mode = "a+"
        else:
            mode = "w"

        with open(path.joinpath(filename), mode, newline="") as f:
            w = DataclassWriter(f, lst_solved_summary, SolvedSummary)
            w.write(skip_header=is_append)

    def write_numpy_as_csv(self, filename: str, data: np.ndarray, path: Path):
        """numpy のデータをcsvファイルに書き出す

        Args:
            filename: ファイル名. `.csv` がなくともメソッドの中でつけるので問題ない
            data: 書き込み対象の numpy データ
            path: 書き込み先のpath
        """
        fullpath_filename = str_util.add_suffix_csv(f"{path}{filename}")
        np.savetxt(fullpath_filename, data, delimiter=",")
        logger.info(f"{fullpath_filename} is written.")

    def write_variables_by_iteration(self, aSolvedDetail: SolvedDetail, path: Path):
        """変数の反復列を出力"""
        if len(aSolvedDetail.lst_variables_by_iter) > 0:
            variables = np.stack([np.concatenate([v.x, v.y, v.s]) for v in aSolvedDetail.lst_variables_by_iter])
            self.write_numpy_as_csv("variables", variables, path)
