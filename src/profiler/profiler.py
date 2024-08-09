"""プロファイリングを行うためのモジュール"""

import cProfile
import pstats
from pathlib import Path

from ..logger import get_main_logger
from ..utils.str_util import add_suffix

logger = get_main_logger()


def profile_decorator(func, path_profile: Path, filename: str, *args, **kwargs):
    """ある関数において実行結果をプロファイルする"""
    full_filename = path_profile.joinpath(add_suffix(filename, ".prof"))
    logger.info(f"Start profiling {full_filename}.")
    pr = cProfile.Profile()
    pr.runcall(func, *args, **kwargs)
    pr.dump_stats(full_filename)
    logger.info(f"End profiling {full_filename}.")


def output_profile_result(path_profile: Path, filename: str):
    """プロファイル結果を確認"""
    sts = pstats.Stats(path_profile.joinpath(f"{filename}"))
    sts.strip_dirs().sort_stats("cumtime").print_stats(30)
