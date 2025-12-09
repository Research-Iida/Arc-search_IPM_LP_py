import csv
from pathlib import Path

import numpy as np

from ..logger import get_main_logger
from ..solver.repository import ISolvedDataRepository
from ..solver.solved_data import SolvedDetail, SolvedSummary
from ..utils import file_util, str_util
from .path_generator import PathGenerator

logger = get_main_logger()


class SolvedDataRepository(ISolvedDataRepository):
    def __init__(self, path_generator: PathGenerator):
        self.path_generator = path_generator

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
        filepath = path / filename

        # 書き込みモード
        mode = "a+" if is_append else "w"

        # ヘッダを書くかどうか
        write_header = not is_append or not filepath.exists()

        with open(filepath, mode, newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SolvedSummary.model_fields.keys())

            if write_header:
                writer.writeheader()

            for solved_summary in lst_solved_summary:
                row = solved_summary.model_dump()
                writer.writerow(row)

    def write_variables_by_iteration(self, aSolvedDetail: SolvedDetail):
        """変数の反復列を出力"""
        summary = aSolvedDetail.aSolvedSummary
        path_result_by_problem_solver_config = self.path_generator.generate_path_result_by_date_problem_solver_config(
            summary.problem_name, summary.solver_name, summary.config_section
        )

        # 書き込むものがなければ終了
        if len(aSolvedDetail.lst_variables_by_iter) == 0:
            return

        variables = np.stack([np.concatenate([v.x, v.y, v.s]) for v in aSolvedDetail.lst_variables_by_iter])
        # self.write_numpy_as_csv("variables", variables, path_result_by_problem_solver_config)
        filename = str_util.add_suffix_csv("variables")
        fullpath_filename = path_result_by_problem_solver_config.joinpath(filename)
        np.savetxt(fullpath_filename, variables, delimiter=",")
        logger.info(f"{fullpath_filename} is written.")
