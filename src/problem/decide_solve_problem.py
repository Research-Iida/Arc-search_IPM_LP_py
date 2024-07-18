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
# Kennington によるサイズの大きい問題群. 普段は skip の対象
kennington_problems = {
    "CRE-A",
    "CRE-B",
    "CRE-C",
    "CRE-D",
    "KEN-07",
    "KEN-11",
    "KEN-13",
    "KEN-18",
    "OSA-07",
    "OSA-14",
    "OSA-30",
    "OSA-60",
    "PDS-06",
    "PDS-10",
    "PDS-20",
}


class TargetProblemError(Exception):
    pass


def decide_solved_problems(
    aLPRepository: ILPRepository,
    num_problem: int | None = None,
    start_problem_number: int | None = None,
    use_kennington: bool = False,
) -> list[str]:
    """解く対象の問題を決める

    Args:
        num_problem: 解く問題数. 与えられていなければデータとして存在するすべての問題を対象にする
        use_kennington: kennington の問題を計算対象の問題とするか. 普段はしない.

    Returns:
        list[str]: 解く対象となった問題名のリスト
    """
    # すべての問題の読み込み
    all_problem_names = set(aLPRepository.get_problem_names())
    if not all_problem_names:
        msg = "There are no problem files! Did you open .tar file?"
        logger.exception(msg)
        raise TargetProblemError(msg)

    # 対象の問題群を決定
    if use_kennington:
        logger.info("Using Kennington problems.")
        skip_problems_in_names = skip_problems & kennington_problems
        problem_names = sorted(list(kennington_problems - skip_problems_in_names))
    else:
        # skip 対象の問題を除外
        skip_problems_in_names = (skip_problems | kennington_problems) & all_problem_names
        problem_names = sorted(list(all_problem_names - skip_problems_in_names))

    for skip_problem in skip_problems_in_names:
        logger.info(f"{skip_problem} is skipped.")

    # 問題番号の決定
    if num_problem is None:
        num_problem = len(problem_names)
    if start_problem_number is None:
        start_problem_number = 0
    end_problem_number = min(start_problem_number + num_problem, len(problem_names))
    if start_problem_number > end_problem_number:
        msg = f"start_problem_number {start_problem_number} is too large! Must be smaller than {end_problem_number}"
        logger.exception(msg)
        raise TargetProblemError(msg)

    return problem_names[start_problem_number:end_problem_number]
