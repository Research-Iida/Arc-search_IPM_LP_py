import sys
import argparse

import numpy as np

from .utils import config_utils
from .logger import get_main_logger, setup_logger
from .run_utils.get_solvers import get_solvers

from .run_utils.define_paths import path_solved_result_by_date
from .run_utils.generate_problem import generate_problem
from .run_utils.solve_problem import optimize
from .run_utils.write_files import write_result_by_problem_solver_config

logger = get_main_logger()


def main(n: int, m: int, solver_name: str | None, config_section: str | None, random_seed: int | None):
    """main 関数
    """
    postfix_random_seed = ""
    if random_seed is not None:
        np.random.seed(random_seed)
        postfix_random_seed = f"_random_seed_{random_seed}"

    # 直接実行された場合ファイルに起こす必要があるため, 新たにlogger設定
    log_file_name = f"solve_generated_problem_n_{n}_m_{m}{postfix_random_seed}"
    if solver_name is not None:
        log_file_name = f"{log_file_name}_{solver_name}"
    if config_section is not None:
        log_file_name = f"{log_file_name}_{config_section}"
    setup_logger(log_file_name)

    config = config_utils.read_config(section=config_section)
    path_result_date = path_solved_result_by_date(config.get("PATH_RESULT"))
    path_result = f"{path_result_date}generated_problem/n_{n}_m_{m}{postfix_random_seed}/"

    problem, opt_sol = generate_problem(n, m)

    # ソルバーごとに解く
    for solver in get_solvers(solver_name, config_section):
        aSolvedDetail = optimize(problem, solver)
        if not aSolvedDetail.v.isclose(opt_sol, threshold=10**(-3)):
            logger.warning(f"Isn't close solution! opt: {opt_sol}, sol: {aSolvedDetail.v}")
        write_result_by_problem_solver_config(aSolvedDetail, path_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", default=10, help="variable size")
    parser.add_argument("-m", default=8, help="constraint number")
    parser.add_argument("-s", "--solver", default=None, help="solver for solving problem")
    parser.add_argument("-c", "--config_section", type=str, default=None, help="config section for solving problem")
    parser.add_argument("-rs", "--random_seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    try:
        main(args.n, args.m, args.solver, args.config_section, args.random_seed)
    except: # NOQA
        logger.exception(sys.exc_info())
