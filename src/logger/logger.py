"""logger module

`config/logging.conf` からロガーに関する情報を読み込む.
ファイルに出力する際は `{日付}_{指定した名前}.log` という形になるよう設定
"""

import glob
import logging
import logging.config
import os
from datetime import date
from pathlib import Path

from ..utils import config_utils, file_util
from ..utils.str_util import add_suffix_log

config_ini = config_utils.read_config()
path_config = Path(config_ini.get("PATH_CONFIG"))
name_logging_conf = config_ini.get("CONFIG_LOGGING")
logging.config.fileConfig(path_config.joinpath(name_logging_conf))
path_log = Path(config_ini.get("PATH_LOG"))


def get_main_logger():
    """`logging.conf` で設定されている MainLogger を出力

    実際に実行しないスクリプト（import されるのみ）であればこのメソッドからloggerを取得する
    ファイルに書き込む場合は `setup_logger` メソッドを使用
    """
    return logging.getLogger("MainLogger")


def setup_logger(
    name: str,
    *,
    batch_date: date = date.today(),
    path_log: Path = path_log,
):
    """logger のセットアップ

    Args:
        name: 出力するlogファイルの名前
        batch_date: バッチ日. default は実行した当日
        path_log: log ファイルの出力先ディレクトリ
    """
    # logger の取得
    logger = logging.getLogger("MainLogger")

    # logファイルの出力名設定
    str_batch_date = batch_date.strftime("%Y%m%d")
    logfile = add_suffix_log(str(path_log.joinpath(f"{str_batch_date}_{name}")))
    # もし既に同じ名前のファイルがある場合は書き直し
    if file_util.exists_file(logfile):
        file_util.remove_files_and_dirs([logfile])

    # create file handler which logs
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter(" %(asctime)s : %(filename)s:%(lineno)s [%(levelname)s] %(message)s")
    fh.setFormatter(fh_formatter)

    # add the handlers to the logger
    logger.addHandler(fh)


def idx_date_start_end_from_filename(path_log: str):
    """ログファイルまでのパスも含めたファイル名から日付の部分を取り出すために,
    日付部分のインデックスを出力

    日付は `%Y%m%d` の形式のため, 8文字分
    """
    start_date_idx = len(path_log)
    end_date_idx = start_date_idx + 8
    return start_date_idx, end_date_idx


def remove_log_files(end_date_to_be_deleted: str, path_log: str = path_log):
    """古くなったログファイルの削除

    Args:
        end_date_to_be_deleted: この日付以前のlogファイルを削除する
        path_log: 削除対象のディレクトリ
    """
    # 存在するログファイルのリスト化
    file_list = glob.glob(f"{path_log}*.log")

    for filename in file_list:
        # 日付は8桁で設定されているため, ファイル名から日付のみ抽出
        start_idx, end_idx = idx_date_start_end_from_filename(path_log)
        file_date = filename[start_idx:end_idx]

        # ファイルの日付が削除対象日以前であれば削除
        if end_date_to_be_deleted >= file_date:
            os.remove(filename)
