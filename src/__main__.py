import argparse
import shutil
import sys
from datetime import date
from pathlib import Path

from .drawer import Drawer
from .infra.path_generator import PathGenerator
from .infra.repository_problem import LPRepository
from .infra.repository_solved_data import SolvedDataRepository
from .logger import get_main_logger, setup_logger
from .problem.repository import ILPRepository
from .profiler.profiler import profile_decorator
from .slack.slack import get_slack_api
from .solver.get_solvers import get_solver, get_solvers
from .solver.solve_problem import solve, solve_and_write
from .utils import config_utils, str_util

logger = get_main_logger()
setup_logger(__name__)
aSlack = get_slack_api()

# スキップする問題群. 基本的にサイズがでかすぎて解けなかったもの
skip_problems = {
    "BLEND",  # SIFファイルに問題があり読み込みできなかった
    "CRE-B",
    "CRE-D",
    "DEGEN2",  # 山下研のサーバーだと実行できなかった
    "DFL001",  # SIF ファイルに問題があり読み込みできなかった
    "E226",  # SIF ファイルに問題があり読み込みできなかった
    "FORPLAN",  # SIFファイルに問題があり読み込みできなかった
    "GFRD-PNC",
    "GROW7",
    "GROW15",
    "GROW22",  # SIFファイルに問題があり読み込みできなかった
    "KEN-11",  # 前処理の途中で落ちた
    "KEN-13",
    "KEN-18",
    "NESM",  # 前処理で実行不可能と判断
    "OSA-30",  # Mps から読み込むのが時間かかりすぎ
    "OSA-60",  # Mps から読み込むのが時間かかりすぎ
    "PDS-06",  # Mps から読み込むのが時間かかりすぎ
    "PDS-10",  # Mps から読み込むのが時間かかりすぎ
    "PDS-20",  # Mps から読み込むのが時間かかりすぎ
    "SCORPION",  # 初期点の計算時に特異行列が出てしまう
    "SIERRA",  # 文字列が数値の所に入っているらしい
    "STOCFOR3",  # 詳細は netlib の README 参照
}
# 解けるサイズではあるものの時間がかかるもの
skip_problems = skip_problems | {
    "80BAU3B",
    "FIT2D",
    "FIT2P",
    "OSA-07",
    "OSA-14",
    "QAP15",
}

# 出力されるファイル名
str_today = date.today().strftime("%Y%m%d")
name_result = str_util.add_suffix_csv(f"{str_today}_result")
# log に日付を入れるためのメッセージ
msg_for_logging_today = f"[{str_today}] "


class TargetProblemError(Exception):
    pass


def decide_solved_problems(
    aLPRepository: ILPRepository,
    num_problem: int | None = None,
    start_problem_number: int | None = None,
) -> list[str]:
    """解く対象の問題を決める

    Args:
        num_problem: 解く問題数. 与えられていなければデータとして存在するすべての問題を対象にする

    Returns:
        list[str]: 解く対象となった問題名のリスト
    """
    # すべての問題の読み込み
    all_problem_files = set(aLPRepository.get_problem_names())
    if not all_problem_files:
        msg = "There are no problem files! Did you open .tar file?"
        logger.exception(msg)
        raise TargetProblemError(msg)

    # skip 対象の問題を除外
    skip_problems_in_files = skip_problems & all_problem_files
    problem_files = sorted(list(all_problem_files - skip_problems_in_files))
    for skip_problem in skip_problems_in_files:
        logger.info(f"{skip_problem} is skipped.")

    # 問題番号の決定
    if num_problem is None:
        num_problem = len(problem_files)
    if start_problem_number is None:
        start_problem_number = 0
    end_problem_number = min(start_problem_number + num_problem, len(problem_files))
    if start_problem_number > end_problem_number:
        msg = f"start_problem_number {start_problem_number} is too large! Must be smaller than {end_problem_number}"
        logger.exception(msg)
        raise TargetProblemError(msg)

    return problem_files[start_problem_number:end_problem_number]


def copy_optimization_parameters(path_result: Path, config_section: str):
    """`config_optimizer.ini` を結果を格納するディレクトリにコピー

    Args:
        path_result (Path): 結果を書き込む先のディレクトリ
    """
    config = config_utils.read_config(section=config_section)
    path_config = config.get("PATH_CONFIG")
    name_config_opt = config.get("CONFIG_OPTIMIZER")
    origin_config_opt = f"{path_config}{name_config_opt}"
    destination_config_opt = path_result.joinpath(name_config_opt)
    logger.info(f"Write {origin_config_opt} to {destination_config_opt}")
    shutil.copyfile(origin_config_opt, destination_config_opt)


def main(
    num_problem: int | None,
    name_solver: str | None,
    config_section: str | None,
    start_problem_number: int | None = None,
):
    """main関数


    Args:
        num_problem: 求解する問題数. 指定がなければすべての問題
        name_solver: 使用するソルバー. 指定がなければスクリプト内で指定したすべてのアルゴリズムで実行する
        config_section: 使用する config のセクション. 指定がなければスクリプト内で指定したすべてのセクションで実行する
        start_problem_number: 整列した問題ファイルの中から指定された問題番号以降を解く
    """
    if start_problem_number is None or start_problem_number <= 0:
        start_problem_number = 0

    if num_problem is None:
        msg_solving_benchmarks = "solving all NETLIB benchmarks."
    else:
        msg_solving_benchmarks = f"solving {num_problem} NETLIB benchmarks from {start_problem_number}th problem."
    msg = f"{msg_for_logging_today}Start {msg_solving_benchmarks}"
    logger.info(msg)
    aSlack.notify(msg)

    # 各種インスタンスの用意
    aLPRepository = LPRepository(config_section)
    aSolvedDataRepository = SolvedDataRepository(config_section)

    # 対象の問題の決定
    problem_files = decide_solved_problems(aLPRepository, num_problem, start_problem_number)
    target_problem_number = len(problem_files)
    logger.info(f"Target problems number: {target_problem_number}")

    # 書き込み先のディレクトリを作成
    path_generator = PathGenerator(config_section=config_section)
    path_result = path_generator.generate_path_result_by_date()
    # パラメータもコピーしておく
    copy_optimization_parameters(path_result, config_section)

    # csvのヘッダーを書き出す
    aSolvedDataRepository.write_SolvedSummary([], name_result, path=path_result)

    # 並列処理の設定
    # max_cpu_core = os.cpu_count() - 1

    # 計算して結果をファイルに記載
    for idx, filename in enumerate(problem_files):
        sum_probelm_idx = idx + start_problem_number
        solving_msg = f"solving {idx + 1}/{target_problem_number} problem: {filename} (sum idx: {sum_probelm_idx})"
        msg = f"{msg_for_logging_today}Start {solving_msg}"
        logger.info(msg)
        aSlack.notify(msg)

        # 最初にline search で解いてキャッシュに入れる
        _ = solve(filename, get_solver("line", config_utils.test_section), aLPRepository)

        # ソルバーごとに解く. 毎回初期化した方が都合がいいので for 構文の中で取り出す
        for solver in get_solvers(name_solver, config_section):
            aSolvedDetail = solve_and_write(
                filename, solver, aLPRepository, aSolvedDataRepository, name_result, path_result
            )
            aSolvedDataRepository.write_variables_by_iteration(aSolvedDetail)

            summary = aSolvedDetail.aSolvedSummary
            path_result_by_problem_solver_config = path_generator.generate_path_result_by_date_problem_solver_config(
                summary.problem_name, summary.solver_name, summary.config_section
            )
            Drawer(path_result_by_problem_solver_config).run(aSolvedDetail)

        # 並列処理: メモリが爆発して逆に遅くなるためやらないほうがいい
        # from multiprocessing import Process
        # process_list = []
        # for solver in lst_solver:
        #     kwargs = {
        #         "filename": filename, "solver": solver, "aLPRepository": aLPRepository, "aSolvedDataRepository": aSolvedDataRepository,
        #         "name_result": name_result, "path_result": path_result
        #     }
        #     process = Process(target=solve_and_write, kwargs=kwargs)
        #     process.start()
        #     process_list.append(process)
        # for process in process_list:
        #     process.join()

        # 何番目の処理が終わったか
        msg = f"{msg_for_logging_today}End {solving_msg}"
        logger.info(msg)
        aSlack.notify(msg)

    msg = f"{msg_for_logging_today}End {msg_solving_benchmarks}"
    logger.info(msg)
    aSlack.notify(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_problem", help="Number of problems to solve", type=int, default=None)
    parser.add_argument("-s", "--solver", default=None, help="solver for solving problem")
    parser.add_argument("-c", "--config_section", type=str, default=None, help="config section for solving problem")
    parser.add_argument("-sn", "--start_problem_number", type=int, default=None, help="start problem from this number")
    args = parser.parse_args()

    try:
        profile_decorator(
            main, "solve_all_problems", args.num_problem, args.solver, args.config_section, args.start_problem_number
        )
        aSlack.notify_mentioned(f"{msg_for_logging_today}End calculation.")
    except:  # NOQA
        aSlack.notify_error()
        logger.exception(sys.exc_info())
