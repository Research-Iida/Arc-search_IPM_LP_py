from configparser import SectionProxy
from datetime import date
from pathlib import Path

from ..utils.config_utils import read_config
from ..utils.file_util import create_dir_if_not_exists


class PathGenerator:
    """config をもとにローカルで必要なパスの作成に責務を持つクラス"""

    config: SectionProxy
    str_date: str

    def __init__(self, config_section: str, date_: date | None = None):
        """初期化

        Args:
            config_section (str): config から読みだす際のセクション名
            date (date): 計算実行日などの日付. 出力するパスに日付が入るので初期化時に指定しておく（でないと日付を超えた際に出力パスが異なる）. 入力がなければ今日の日付
        """
        self.config = read_config(section=config_section)

        if date_ is None:
            self.str_date = date.today().strftime("%Y%m%d")
        else:
            self.str_date = date_.strftime("%Y%m%d")

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
        result = self.generate_path_result().joinpath(self.str_date)
        create_dir_if_not_exists(result)
        return result

    def generate_path_result_by_date_problem(self, problem_name: str) -> Path:
        """最適化の結果を問題ごとに変えるため, 対応するためのルール

        Args:
            path_result (Path): ベースとなる書き込み先
        """
        result = self.generate_path_result_by_date().joinpath(problem_name)
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
        create_dir_if_not_exists(result)
        return result

    def generate_path_config_optimizer(self) -> Path:
        return Path(self.config.get("PATH_CONFIG")).joinpath(self.config.get("CONFIG_OPTIMIZER"))
