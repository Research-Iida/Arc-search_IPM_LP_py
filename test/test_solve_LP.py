import pytest

from src.utils.config_utils import read_config
from src.utils.file_util import remove_files_and_dirs
from src import solve_netlib

config_section = "TEST"
# テスト対象のアルゴリズム一覧
target_algorithms: list[str] = [
    "line",
    "arc",
    "arc_restarting",
    "arc_restarting_proven",
]
config = read_config(section=config_section)


class TestSolveKB2:
    name_problem = "KB2"

    @pytest.fixture(scope="class")
    def remove_KB2_processed(self):
        """KB2を解いた後に `data/test/processed` ディレクトリに作成されるファイルを削除
        読み込みが processed の csv からになってしまうため
        """
        yield

        path_processed = config.get("PATH_PROCESSED")
        for name_file in [f"{path_processed}{self.name_problem}_{i}.csv" for i in ["A", "b", "c"]]:
            remove_files_and_dirs([name_file])

    @pytest.mark.parametrize("algorithm", target_algorithms)
    def test_solve_KB2_with_all_algorithms(self, algorithm, remove_KB2_processed):
        """`data/test/raw/netlib/KB2.mps` から問題を読み込み,
        TEST セクションの設定でソルバーを使用して最適化する
        """
        solve_netlib.main(self.name_problem, algorithm, config_section)
