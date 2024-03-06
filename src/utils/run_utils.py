
from datetime import date

import numpy as np

from ..logger import get_main_logger
from ..data_access import CsvHandler
from ..problem import LinearProgrammingProblemStandard as LPS
from ..solver import LPSolver
from ..solver.solver import LPVariables, SolvedDetail
from ..drawer import Drawer
from .file_util import create_dir_if_not_exists

logger = get_main_logger()


def deco_logging(problem_name: str, solver: LPSolver):
    """求解開始をログ, および slack に出力するデコレータ

    Args:
        problem_name: 求解対象の問題の名前
        solver: 求解する際の solver
    """
    solver_name = solver.__class__.__name__
    config_section = solver.config_section

    def _deco_logging(func):
        def wrapper(*args, **kwargs):
            msg_prefix = f"[{solver_name}] [{config_section}]"
            msg_start = f"{msg_prefix} Start solving {problem_name}."
            logger.info(msg_start)

            output = func(*args, **kwargs)

            msg_end = f"{msg_prefix} End solving {problem_name}."
            logger.info(msg_end)
            return output
        return wrapper
    return _deco_logging


def optimize(aLP: LPS, aLPSolver: LPSolver, v_0: LPVariables | None = None) -> SolvedDetail:
    """最適化の実行. ロギングなども同時に行う

    Args:
        aLP: 求解対象の線形計画問題
        aLPSolver: 線形計画問題の solver
        v_0: 初期点

    Returns:
        求解した結果
    """
    problem_name = aLP.name
    solver_name = aLPSolver.__class__.__name__

    # 入力をデコレータに渡すための実質のmain関数
    @deco_logging(problem_name, aLPSolver)
    def _optimize():
        return aLPSolver.run(aLP, v_0)

    output = _optimize()
    # 求解できなかったら warning
    if not output.aSolvedSummary.is_solved:
        msg = f"[{solver_name}] [{aLPSolver.config_section}] Algorithm cannot solve {problem_name}!"
        logger.warning(msg)
    return output


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


def write_result_by_problem_solver_config(aSolvedDetail: SolvedDetail, path_result: str):
    """計算に関わるいろいろな設定を書き込む

    Args:
        path_result: result ディレクトリ. この下に `問題名/ソルバー名/セクション名` というディレクトリを作成して書き込みを行う
    """
    summary = aSolvedDetail.aSolvedSummary
    path_result_by_problem = path_solved_result_by_problem(path_result, summary.problem_name)
    path_result_by_problem_solver_config = path_solved_result_by_solver_with_config(path_result_by_problem, summary.solver_name, summary.config_section)

    # 変数の反復列をcsvで出力
    if len(aSolvedDetail.lst_variables_by_iter) > 0:
        variables = np.stack([np.concatenate([v.x, v.y, v.s]) for v in aSolvedDetail.lst_variables_by_iter])
        CsvHandler().write_numpy("variables", variables, path_result_by_problem_solver_config)

    # グラフ描画
    Drawer(path_result_by_problem_solver_config).run(aSolvedDetail)
