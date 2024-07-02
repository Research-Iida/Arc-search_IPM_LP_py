from dataclass_csv import DataclassWriter

from ..solver.solver import SolvedSummary
from ..utils import file_util, str_util


class SolvedDataRepository:
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
