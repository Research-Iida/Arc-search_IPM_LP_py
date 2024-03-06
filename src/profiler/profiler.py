"""プロファイリングを行うためのモジュール"""
import cProfile
import pstats

from ..logger import get_main_logger
from ..utils import config_utils

logger = get_main_logger()
config_ini = config_utils.read_config()
path_profile = config_ini.get("PATH_PROFILE")


def profile_decorator(func, filename: str, *args, **kwargs):
    """ある関数において実行結果をプロファイルする
    """
    full_filename = f"{path_profile}{filename}.prof"
    logger.info(f"Start profiling {full_filename}.")
    pr = cProfile.Profile()
    pr.runcall(func, *args, **kwargs)
    pr.dump_stats(full_filename)
    logger.info(f"End profiling {full_filename}.")


def output_profile_result(filename: str):
    """プロファイル結果を確認"""
    sts = pstats.Stats(f"{path_profile}{filename}")
    sts.strip_dirs().sort_stats("cumtime").print_stats(30)
