import sys
import argparse
from typing import Optional

import numpy as np

from .utils import config_utils
from .slack import Slack
from .logger import get_main_logger, setup_logger
from .solver import get_solvers
from .solver.solver import LPVariables
from .problem import LinearProgrammingProblemStandard as LPS
from .utils.run_utils import path_solved_result_by_date, optimize, write_result_by_problem_solver_config


logger = get_main_logger()
aSlack = Slack()


def generate_problem(n: int, m: int) -> tuple[LPS, LPVariables]:
    """変数サイズと制約数を与えて LP を作成

    Args:
        n (int): 変数サイズ
        m (int): 制約数

    Returns:
        LPS: 問題
        LPVariables: 最適解
    """
    mask = [1 if ind < m else 0 for ind in range(n)]
    np.random.shuffle(mask)

    # 最適解の決定
    opt_x = np.multiply(np.random.rand(n), mask)
    opt_s = np.random.rand(n)
    opt_s = opt_s - np.multiply(opt_s, mask)
    opt_y = np.random.rand(m) - 0.5

    A = np.random.rand(m, n) - 0.5
    b = np.matmul(A, opt_x)
    c = np.matmul(A.T, opt_y) + opt_s

    return LPS(A, b, c), LPVariables(opt_x, opt_y, opt_s)


def main(n: int, m: int, solver_name: Optional[str], config_section: Optional[str]):
    """main 関数
    """
    # 直接実行された場合ファイルに起こす必要があるため, 新たにlogger設定
    log_file_name = f"solve_generated_problem_n_{n}_m_{m}"
    if solver_name is not None:
        log_file_name = f"{log_file_name}_{solver_name}"
    if config_section is not None:
        log_file_name = f"{log_file_name}_{config_section}"
    setup_logger(log_file_name)

    config = config_utils.read_config(section=config_section)
    path_result = path_solved_result_by_date(config.get("PATH_RESULT"))

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
    parser.add_argument("-m", default=4, help="constraint number")
    parser.add_argument("-s", "--solver", default=None, help="solver for solving problem")
    parser.add_argument("-c", "--config_section", type=str, default=None, help="config section for solving problem")
    parser.add_argument("-rs", "--random_seed", type=int, default=None, help="random seed")
    args = parser.parse_args()

    if args.random_seed is not None:
        np.random.seed(args.random_seed)

    try:
        main(args.n, args.m, args.solver, args.config_section)
    except: # NOQA
        aSlack.notify_error()
        logger.exception(sys.exc_info())
