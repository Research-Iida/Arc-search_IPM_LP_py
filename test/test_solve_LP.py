import glob

import pytest

from src import solve_netlib
from src.infra.path_generator import PathGenerator
from src.utils.config_utils import test_section
from src.utils.file_util import remove_files_and_dirs

# テスト対象のアルゴリズム一覧
target_algorithms: list[str] = [
    "line",
    "arc",
    "arc_restarting",
    "arc_restarting_proven",
    "inexact_arc",
    "inexact_line",
    "iterative_refinement",
]


class TestSolveKB2:
    name_problem = "KB2"

    @pytest.fixture(scope="class")
    def remove_KB2_processed(self):
        """KB2を解く前に `data/test/processed` ディレクトリに作成されるファイルを削除"""
        path_processed = PathGenerator(test_section).generate_path_data_processed()
        remove_files_and_dirs(glob.glob(f"{path_processed}/{self.name_problem}*"))

        yield

    @pytest.mark.parametrize("algorithm", target_algorithms)
    def test_solve_KB2_with_all_algorithms(self, algorithm, remove_KB2_processed):
        """`data/test/raw/netlib/KB2.mps` から問題を読み込み,
        TEST セクションの設定でソルバーを使用して最適化する
        """
        solve_netlib.main(self.name_problem, algorithm, test_section)
