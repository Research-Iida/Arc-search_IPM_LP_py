import sys
import argparse
from typing import Optional

from .utils import config_utils
from .data_access import CsvHandler, MpsLoader
from .logger import get_main_logger, setup_logger
from .problem import LPPreprocessor, LinearProgrammingProblemStandard

logger = get_main_logger()


def preprocess(
    problem_name: str, aMpsLoader: MpsLoader, aCsvHandler: CsvHandler
) -> LinearProgrammingProblemStandard:
    """前処理を施し, 標準形となった Netlib LP を csv で書き込む
    """
    logger.info(f"Start loading problem '{problem_name}'")
    aLP_origin = aMpsLoader.run(problem_name).convert_standard()
    logger.info("End loading.")
    logger.info(f"Origin dimension: n: {aLP_origin.n}, m: {aLP_origin.m}")
    logger.info("Start preprocessing.")
    aLP = LPPreprocessor().run(aLP_origin)
    logger.info("End preprocessing.")
    logger.info("Start writing csv.")
    aCsvHandler.write_LP(aLP, problem_name)
    logger.info("End writing.")
    return aLP


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
