"""実行されたプロファイリングの結果を展開する関数

>>> poetry run python -m src.profile_result {対称のファイル名}
"""

import sys

from .profiler.profiler import output_profile_result


if __name__ == "__main__":
    """標準入力から実行された際は出力結果を確認

    `{標準入力された文字列}.prof` ファイルの結果を読み込む
    """
    prof_file_name = sys.argv[1]
    output_profile_result(prof_file_name)
