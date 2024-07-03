from datetime import date
from pathlib import Path

from ..logger import get_main_logger
from ..utils.file_util import create_dir_if_not_exists

logger = get_main_logger()


def path_solved_result_by_date(path_result: Path) -> Path:
    """最適化の結果を格納するディレクトリを日付ごとに変えるため, 対応するためのルール

    Args:
        path_result (Path): ベースとなる書き込み先

    Returns:
        Path: `{path_result}{実行日の日付, YYYYMMDD}`. 対応するディレクトリが作成された状態になる
    """
    str_today = date.today().strftime("%Y%m%d")
    result = path_result.joinpath(str_today)
    logger.info(f"Results will be written to {result}")
    create_dir_if_not_exists(result)
    return result


def path_solved_result_by_problem(path_result: Path, problem_name: str) -> Path:
    """最適化の結果を問題ごとに変えるため, 対応するためのルール

    Args:
        path_result (Path): ベースとなる書き込み先
    """
    result = path_result.joinpath(problem_name)
    logger.info(f"Results will be written to {result}")
    create_dir_if_not_exists(result)
    return result


def path_solved_result_by_solver_with_config(path_result: Path, solver_name: str, config_section: str) -> Path:
    """最適化の結果をソルバーごとに書き込み先を変えるため, 対応するためのルール

    Args:
        path_result (Path): ベースとなる書き込み先. 日付ごとにディレクトリが変化しても大丈夫
    """
    result = path_result.joinpath(solver_name, config_section)
    logger.info(f"Results will be written to {result}")
    create_dir_if_not_exists(result)
    return result
