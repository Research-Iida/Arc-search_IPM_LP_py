import argparse
import sys

from .infra.path_generator import PathGenerator
from .infra.python.repository_problem import LPRepository
from .logger import get_main_logger, setup_logger
from .slack.slack import get_slack_api
from .solver.solve_problem import preprocess

logger = get_main_logger()
aSlack = get_slack_api()


def main(problem_name: str, config_section: str | None, with_julia: bool):
    """main 関数"""
    aSlack.notify(f"Start preprocessing '{problem_name}'")
    # 直接実行された場合ファイルに起こす必要があるため, 新たにlogger設定
    log_file_name = f"preprocess_{problem_name}"
    if config_section is not None:
        log_file_name = f"{log_file_name}_{config_section}"
    setup_logger(log_file_name)

    if with_julia:
        logger.info("Using julia for netlib loading.")
        from .infra.julia.repository_problem import JuliaLPRepository

        repository = JuliaLPRepository(PathGenerator(config_section))
    else:
        repository = LPRepository(PathGenerator(config_section))

    preprocess(problem_name, repository)

    aSlack.notify_mentioned(f"End preprocessing '{problem_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", help="display a given problem name")
    parser.add_argument("-c", "--config_section", type=str, default=None, help="config section for solving problem")
    parser.add_argument("-j", "--with_julia", help="load using julia QPSReader", action="store_true")
    args = parser.parse_args()

    try:
        main(args.problem_name, args.config_section, args.with_julia)
    except:  # NOQA
        logger.exception(sys.exc_info())
