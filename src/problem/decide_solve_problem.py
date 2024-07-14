from ..logger import get_main_logger
from .repository import ILPRepository

logger = get_main_logger()

# スキップする問題群
skip_problems = {
    "BLEND",  # SIFファイルに問題があり読み込みできなかった
    "DFL001",  # SIFファイルに問題があり読み込みできなかった
    "FORPLAN",  # SIFファイルに問題があり読み込みできなかった
    "GFRD-PNC",  # SIFファイルに問題があり読み込みできなかった
    "GREENBEB",  # CGだと永遠に終わらない
    "KEN-18",  # exact に線形方程式を解くにはサイズがでかすぎる
    "OSA-60",  # exact に線形方程式を解くにはサイズがでかすぎる
    "PDS-20",  # exact に線形方程式を解くにはサイズがでかすぎる
    "SCORPION",  # 初期点の計算時に特異行列が出てしまう
    "SIERRA",  # 文字列が数値の所に入っているらしい
    "STOCFOR3",  # exact に線形方程式を解くにはサイズがでかすぎる
}
# 解けるサイズではあるものの時間がかかるもの
# skip_problems = skip_problems | {
#     "80BAU3B",
#     "CRE-B",
#     "CRE-D",
#     "FIT2D",
#     "FIT2P",
#     "KEN-11",
#     "KEN-13",
#     "KEN-18",
#     "OSA-07",
#     "OSA-14",
#     "QAP15",
#     "OSA-30",
#     "OSA-60",
#     "PDS-06",
#     "PDS-10",
#     "PDS-20",
#     "STOCFOR3",
# }


class TargetProblemError(Exception):
    pass


def decide_solved_problems(
    aLPRepository: ILPRepository,
    num_problem: int | None = None,
    start_problem_number: int | None = None,
) -> list[str]:
    """解く対象の問題を決める

    Args:
        num_problem: 解く問題数. 与えられていなければデータとして存在するすべての問題を対象にする

    Returns:
        list[str]: 解く対象となった問題名のリスト
    """
    # すべての問題の読み込み
    all_problem_files = set(aLPRepository.get_problem_names())
    if not all_problem_files:
        msg = "There are no problem files! Did you open .tar file?"
        logger.exception(msg)
        raise TargetProblemError(msg)

    # skip 対象の問題を除外
    skip_problems_in_files = skip_problems & all_problem_files
    problem_files = sorted(list(all_problem_files - skip_problems_in_files))
    for skip_problem in skip_problems_in_files:
        logger.info(f"{skip_problem} is skipped.")

    # 問題番号の決定
    if num_problem is None:
        num_problem = len(problem_files)
    if start_problem_number is None:
        start_problem_number = 0
    end_problem_number = min(start_problem_number + num_problem, len(problem_files))
    if start_problem_number > end_problem_number:
        msg = f"start_problem_number {start_problem_number} is too large! Must be smaller than {end_problem_number}"
        logger.exception(msg)
        raise TargetProblemError(msg)

    return problem_files[start_problem_number:end_problem_number]
