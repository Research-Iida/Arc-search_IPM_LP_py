"""プロファイリングを行うためのモジュール"""

import cProfile
import pstats
from pathlib import Path

from ..logger import get_main_logger

logger = get_main_logger()


def profile_decorator(func, full_filename: Path, *args, **kwargs):
    """ある関数において実行結果をプロファイルする"""
    logger.info(f"Start profiling {full_filename}.")
    pr = cProfile.Profile()
    pr.runcall(func, *args, **kwargs)
    pr.dump_stats(full_filename)
    logger.info(f"End profiling {full_filename}.")


def output_profile_result(full_filename: Path):
    """プロファイル結果を確認"""
    sts = pstats.Stats(full_filename)
    sts.strip_dirs().sort_stats("cumtime").print_stats(30)
