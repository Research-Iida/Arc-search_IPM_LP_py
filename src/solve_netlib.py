import sys
import argparse
from datetime import date
from typing import Optional

from .utils import config_utils, str_util
from .slack import Slack
from .profiler.profiler import profile_decorator
from .data_access import CsvHandler, MpsLoader
from .logger import get_main_logger, setup_logger
from .solver import LPSolver, get_solvers
from .solver.solver import SolvedDetail
from .utils.run_utils import path_solved_result_by_date, path_solved_result_by_problem, optimize, write_result_by_problem_solver_config
from .preprocess_netlib import preprocess

logger = get_main_logger()


def solve(
    problem_name: str,
    aLPSolver: LPSolver,
    aMpsLoader: MpsLoader,
    aCsvHandler: CsvHandler
) -> SolvedDetail:
    """ベンチマークの問題を読み取り, 解く

    すでに問題を前処理したファイルが存在する場合, そこから読み取ることで時間を短縮する

    Args:
        problem_name: ベンチマーク問題の名前
        aLPSolver: 線形計画問題を解くためのソルバー. 抽象クラスなので, 実際に使用する際は
            ソルバーを指定

    Returns:
        SolvedDetail: 最適化によって作成された諸解群
    """
    # すでに前処理済みの問題であれば, csvファイルから読み込む
    if aCsvHandler.can_read_LP(problem_name):
        logger.info(f"There are preprocessed {problem_name} data.")
        logger.info(f"Read {problem_name} csv files.")
        aLP = aCsvHandler.read_LP(problem_name)
    # そうでなければ前処理を行い, csvファイルに書き込んでおく
    else:
        aLP = preprocess(problem_name, aMpsLoader, aCsvHandler)

    # 最適化
    return optimize(aLP, aLPSolver)


def solve_and_write(
    filename: str, solver: LPSolver, aMpsLoader: MpsLoader, aCsvHandler: CsvHandler,
    name_result: str, path_result: str
) -> SolvedDetail:
    """問題を解き, 結果を格納する. `__main__.py` で使用するので書き出しておく"""
    aSolvedDetail = solve(
        filename, solver,
        aMpsLoader=aMpsLoader, aCsvHandler=aCsvHandler
    )
    # 計算が終わるたびに都度書き込みを行う
    aCsvHandler.write_SolvedSummary(
        [aSolvedDetail.aSolvedSummary],
        name_result,
        path=path_result,
        is_append=True
    )
    return aSolvedDetail


def main(problem_name: str, solver_name: Optional[str], config_section: Optional[str]):
    """main 関数
    """
    # 直接実行された場合ファイルに起こす必要があるため, 新たにlogger設定
    log_file_name = f"solve_{problem_name}"
    if solver_name is not None:
        log_file_name = f"{log_file_name}_{solver_name}"
    if config_section is not None:
        log_file_name = f"{log_file_name}_{config_section}"
    setup_logger(log_file_name)

    config = config_utils.read_config(section=config_section)
    path_result = path_solved_result_by_date(config.get("PATH_RESULT"))
    path_result_by_problem = path_solved_result_by_problem(path_result, problem_name)
    aMpsLoader = MpsLoader(config.get("PATH_NETLIB"))
    aCsvHandler = CsvHandler(config_section)

    # 出力されるファイル名
    str_today = date.today().strftime("%Y%m%d")
    name_result = str_util.add_suffix_csv(f"{str_today}_result")
    # csvのヘッダーを書き出す
    aCsvHandler.write_SolvedSummary([], name_result, path=path_result_by_problem)

    # ソルバーごとに解く
    for solver in get_solvers(solver_name, config_section):
        aSolvedDetail = solve_and_write(problem_name, solver, aMpsLoader, aCsvHandler, name_result, path_result_by_problem)
        write_result_by_problem_solver_config(aSolvedDetail, path_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", help="display a given problem name")
    parser.add_argument("-s", "--solver", default=None, help="solver for solving problem")
    parser.add_argument("-c", "--config_section", type=str, default=None, help="config section for solving problem")
    parser.add_argument("-m", "--mention", action='store_true', help="slack mention flag")
    args = parser.parse_args()

    problem_name = args.problem_name
    profile_name = f"solve_{problem_name}"
    solver_name = args.solver
    if solver_name is not None:
        profile_name = f"{profile_name}_{solver_name}"
    config_section = args.config_section
    if config_section is not None:
        profile_name = f"{profile_name}_{config_section}"

    aSlack = Slack()

    try:
        profile_decorator(main, profile_name, problem_name, solver_name, config_section)
        if args.mention:
            aSlack.notify_mentioned("End calculation.")
    except: # NOQA
        aSlack.notify_error()
        logger.exception(sys.exc_info())
