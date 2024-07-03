from datetime import date
from pathlib import Path

from ..logger import get_main_logger
from ..utils.config_utils import read_config
from ..utils.file_util import create_dir_if_not_exists

logger = get_main_logger()


class PathGenerator:
    """config をもとにローカルで必要なパスの作成に責務を持つクラス"""

    def __init__(self, config_section: str):
        self.config = read_config(section=config_section)

    def generate_path_data(self) -> Path:
        return Path(self.config.get("PATH_DATA"))

    def generate_path_data_raw(self) -> Path:
        return self.generate_path_data().joinpath(self.config.get("PATH_RAW"))

    def generate_path_netlib(self) -> Path:
        return self.generate_path_data_raw().joinpath(self.config.get("PATH_NETLIB"))

    def generate_path_data_processed(self) -> Path:
        return self.generate_path_data().joinpath(self.config.get("PATH_PROCESSED"))

    def generate_path_result(self) -> Path:
        return Path(self.config.get("PATH_RESULT"))

    def generate_path_result_by_date(self) -> Path:
        """最適化の結果を格納するディレクトリを日付ごとに変えるため, 対応するためのルール

        Returns:
            Path: `{path_result}{実行日の日付, YYYYMMDD}`. 対応するディレクトリが作成された状態になる
        """
        str_today = date.today().strftime("%Y%m%d")
        result = self.generate_path_result().joinpath(str_today)
        logger.info(f"Results will be written to {result}")
        create_dir_if_not_exists(result)
        return result

    def generate_path_result_by_date_problem(self, problem_name: str) -> Path:
        """最適化の結果を問題ごとに変えるため, 対応するためのルール

        Args:
            path_result (Path): ベースとなる書き込み先
        """
        result = self.generate_path_result_by_date().joinpath(problem_name)
        logger.info(f"Results will be written to {result}")
        create_dir_if_not_exists(result)
        return result

    def generate_path_result_by_date_problem_solver_config(
        self, problem_name: str, solver_name: str, config_section: str
    ) -> Path:
        """最適化の結果をソルバーごとに書き込み先を変えるため, 対応するためのルール

        Args:
            path_result (Path): ベースとなる書き込み先. 日付ごとにディレクトリが変化しても大丈夫
        """
        result = self.generate_path_result_by_date_problem(problem_name).joinpath(solver_name, config_section)
        logger.info(f"Results will be written to {result}")
        create_dir_if_not_exists(result)
        return result
