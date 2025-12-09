import argparse
import sys

import numpy as np

from .app.get_solvers import get_solvers
from .drawer import Drawer
from .infra.path_generator import PathGenerator
from .infra.repository_solved_data import SolvedDataRepository
from .logger import get_main_logger, setup_logger
from .problem.generate_problem import generate_problem
from .solver.solve_problem import optimize

logger = get_main_logger()


def main(n: int, m: int, solver_name: str | None, config_section: str | None, random_seed: int | None):
    """main 関数"""
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

    path_generator = PathGenerator(config_section=config_section)
    aSolvedDataRepository = SolvedDataRepository(path_generator)

    problem, opt_sol = generate_problem(n, m)

    # ソルバーごとに解く
    for solver in get_solvers(solver_name, config_section):
        aSolvedDetail = optimize(problem, solver)
        if not aSolvedDetail.v.isclose(opt_sol, threshold=10 ** (-3)):
            logger.warning(f"Isn't close solution! opt: {opt_sol}, sol: {aSolvedDetail.v}")
        aSolvedDataRepository.write_variables_by_iteration(aSolvedDetail)

        summary = aSolvedDetail.aSolvedSummary
        path_result_by_problem_solver_config = path_generator.generate_path_result_by_date_problem_solver_config(
            summary.problem_name, summary.solver_name, summary.config_section
        )
        Drawer(path_result_by_problem_solver_config).run(aSolvedDetail)


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
    except:  # NOQA
        logger.exception(sys.exc_info())
