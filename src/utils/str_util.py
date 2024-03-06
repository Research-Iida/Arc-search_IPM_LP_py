from typing import Optional


def is_empty(target: Optional[str]) -> bool:
    if target is None:
        return True
    if len(target) == 0:
        return True
    return False


def add_suffix(filename: str, suffix: str) -> str:
    """ファイル名に指定した接尾辞がついていなければ追加

    Args:
        filename (str): 接尾辞追加対象のファイル名
        suffix (str): 追加対象接尾辞

    Returns:
        str: 接尾辞を追加したファイル名
    """
    if not filename.endswith(suffix):
        filename += suffix
    return filename


def add_suffix_log(filename: str) -> str:
    """ファイル名に `.log` とついていなければ追加"""
    return add_suffix(filename, ".log")


def add_suffix_csv(filename: str) -> str:
    """ファイル名に`.csv` とついていなければ追加"""
    return add_suffix(filename, ".csv")


def add_suffix_zip(filename: str) -> str:
    return add_suffix(filename, ".zip")
