import sys
import argparse
from typing import Optional

from .utils import config_utils
from .data_access import CsvHandler, MpsLoader
from .logger import get_main_logger, setup_logger
from .run_utils.solve_problem import preprocess

logger = get_main_logger()


def main(problem_name: str, config_section: Optional[str]):
    """main 関数
    """
    # 直接実行された場合ファイルに起こす必要があるため, 新たにlogger設定
    log_file_name = f"preprocess_{problem_name}"
    if config_section is not None:
        log_file_name = f"{log_file_name}_{config_section}"
    setup_logger(log_file_name)

    config = config_utils.read_config(section=config_section)
    aMpsLoader = MpsLoader(config.get("PATH_NETLIB"))
    aCsvHandler = CsvHandler(config_section)

    preprocess(problem_name, aMpsLoader, aCsvHandler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem_name", help="display a given problem name")
    parser.add_argument("-c", "--config_section", type=str, default=None, help="config section for solving problem")
    args = parser.parse_args()

    try:
        main(args.problem_name, args.config_section)
    except: # NOQA
        logger.exception(sys.exc_info())
