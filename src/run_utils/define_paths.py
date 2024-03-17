
from datetime import date

from ..logger import get_main_logger
from ..utils.file_util import create_dir_if_not_exists

logger = get_main_logger()


def path_solved_result_by_date(path_result: str) -> str:
    """最適化の結果を格納するディレクトリを日付ごとに変えるため, 対応するためのルール

    Args:
        path_result (str): ベースとなる書き込み先

    Returns:
        str: `{path_result}{実行日の日付, YYYYMMDD}`. 対応するディレクトリが作成された状態になる
    """
    str_today = date.today().strftime("%Y%m%d")
    result = f"{path_result}{str_today}/"
    logger.info(f"Results will be written to {result}")
    create_dir_if_not_exists(result)
    return result


def path_solved_result_by_problem(path_result: str, problem_name: str) -> str:
    """最適化の結果を問題ごとに変えるため, 対応するためのルール

    Args:
        path_result (str): ベースとなる書き込み先
    """
    result = f"{path_result}{problem_name}/"
    logger.info(f"Results will be written to {result}")
    create_dir_if_not_exists(result)
    return result


def path_solved_result_by_solver_with_config(path_result: str, solver_name: str, config_section: str) -> str:
    """最適化の結果をソルバーごとに書き込み先を変えるため, 対応するためのルール

    Args:
        path_result (str): ベースとなる書き込み先. 日付ごとにディレクトリが変化しても大丈夫
    """
    result = f"{path_result}{solver_name}/{config_section}/"
    logger.info(f"Results will be written to {result}")
    create_dir_if_not_exists(result)
    return result
