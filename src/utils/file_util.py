"""ファイルの処理に関する便利ツールをまとめたスクリプト"""

import os
import shutil

from . import str_util


class FileUtilException(Exception):
    pass


def fullpath(file_name: str, dir_name: str = "") -> str:
    if str_util.is_empty(file_name):
        raise FileUtilException("file_name is empty.")

    if str_util.is_empty(dir_name):
        return file_name

    return f"{str_util.add_suffix(dir_name, '/')}{file_name}"


def create_dir_if_not_exists(dir_name: str):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def copy_files(from_dir_name: str, to_dir_name: str):
    if str_util.is_empty(from_dir_name):
        raise FileUtilException("From dir is empty.")

    if str_util.is_empty(to_dir_name):
        raise FileUtilException("To dir is empty.")

    shutil.copytree(from_dir_name, to_dir_name, dirs_exist_ok=True)


def out_text(contents: list[str], file_path: str):
    with open(file_path, "wt", encoding="utf-8") as f:
        for line in contents:
            f.write(f"{line}\n")


def exists(dir_path: str, file_name: str) -> bool:
    return os.path.isfile(f"{dir_path}/{file_name}")


def exists_file(file_path: str) -> bool:
    return os.path.isfile(file_path)


def copy_file(file_path: str, to_path: str):
    shutil.copy(file_path, to_path)


def remove_files_and_dirs(lst_files_and_dirs: list[str]):
    for file_or_dir in lst_files_and_dirs:
        if os.path.isfile(file_or_dir):
            os.remove(file_or_dir)
        elif os.path.isdir(file_or_dir):
            shutil.rmtree(file_or_dir)
