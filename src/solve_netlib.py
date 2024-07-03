import argparse
import sys
from datetime import date
from pathlib import Path

from .drawer import Drawer
from .infra.path_generator import PathGenerator
from .infra.repository_problem import LPRepository
from .infra.repository_solved_data import SolvedDataRepository
from .logger import get_main_logger, setup_logger
from .profiler.profiler import profile_decorator
from .run_utils.get_solvers import get_solvers
from .run_utils.solve_problem import solve_and_write
from .slack.slack import get_slack_api
from .utils import str_util

logger = get_main_logger()
aSlack = get_slack_api()


def main(problem_name: str, solver_name: str | None, config_section: str | None):
    """main 関数"""
    # 直接実行された場合ファイルに起こす必要があるため, 新たにlogger設定
    log_file_name = f"solve_{problem_name}"
    if solver_name is not None:
        log_file_name = f"{log_file_name}_{solver_name}"
    if config_section is not None:
        log_file_name = f"{log_file_name}_{config_section}"
    setup_logger(log_file_name)

    path_generator = PathGenerator(config_section=config_section)
    path_result_by_problem = path_generator.generate_path_result_by_date_problem(problem_name)
    repository = LPRepository(config_section)
    aSolvedDataRepository = SolvedDataRepository(config_section)

    # 出力されるファイル名
    str_today = date.today().strftime("%Y%m%d")
    name_result = str_util.add_suffix_csv(f"{str_today}_result")
    # csvのヘッダーを書き出す
    aSolvedDataRepository.write_SolvedSummary([], name_result, path=Path(path_result_by_problem))

    # ソルバーごとに解く
    for solver in get_solvers(solver_name, config_section):
        aSolvedDetail = solve_and_write(
            problem_name, solver, repository, aSolvedDataRepository, name_result, path_result_by_problem
        )

        aSolvedDataRepository.write_variables_by_iteration(aSolvedDetail)

        summary = aSolvedDetail.aSolvedSummary
        path_result_by_problem_solver_config = path_generator.generate_path_result_by_date_problem_solver_config(
            summary.problem_name, summary.solver_name, summary.config_section
        )
        Drawer(path_result_by_problem_solver_config).run(aSolvedDetail)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", help="display a given problem name")
    parser.add_argument("-s", "--solver", default=None, help="solver for solving problem")
    parser.add_argument("-c", "--config_section", type=str, default=None, help="config section for solving problem")
    args = parser.parse_args()

    problem_name = args.problem_name
    profile_name = f"solve_{problem_name}"
    solver_name = args.solver
    if solver_name is not None:
        profile_name = f"{profile_name}_{solver_name}"
    config_section = args.config_section
    if config_section is not None:
        profile_name = f"{profile_name}_{config_section}"

    try:
        profile_decorator(main, profile_name, problem_name, solver_name, config_section)
        aSlack.notify_mentioned("End calculation")
    except:  # NOQA
        aSlack.notify_error()
        logger.exception(sys.exc_info())
