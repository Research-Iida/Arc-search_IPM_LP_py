import json
from collections.abc import Iterator
from pathlib import Path

from ..solver.solver import ILPSolver
from .algorithm_builder import AlgorithmBuilder


def load_solver_info(path_json: Path) -> dict[str, list[str]]:
    """計算対象とするソルバーと config_section に関する情報を読み込む"""
    return json.loads(path_json.read_text("utf-8"))


class SolverSelectionError(Exception):
    """solver の選択に失敗したときに発生するエラー"""

    pass


def get_solver(algorithm: str, config_section: str) -> ILPSolver:
    """線形計画問題のソルバー取得

    Args:
        solver: 取得したいソルバーの種類名
        config_section: 使用する config のセクション名
    """
    return AlgorithmBuilder(config_section).build(algorithm)


def get_solvers(
    target_algorithms_and_sections: dict[str, list[str]], name_solver: str | None, config_section: str | None
) -> Iterator[ILPSolver]:
    """対象のソルバー群から複数のソルバー取得

    Args:
        target_algorithms_and_sections: 計算対象のアルゴリズムとその config_section を保持する辞書
    """
    # ソルバー名が明示的に与えられなかった場合, 本スクリプトで定義しているリストからすべての solver を取得
    if name_solver is None:
        target_algorithms = target_algorithms_and_sections.keys()
        # config_section が明示的に与えられなかった場合も同様
        if config_section is None:
            return iter(get_solver(a, c) for a in target_algorithms for c in target_algorithms_and_sections[a])
        return iter(get_solver(a, config_section) for a in target_algorithms)

    if name_solver not in target_algorithms_and_sections:
        raise SolverSelectionError(f"solver '{name_solver}' is not found")
    if config_section is None:
        return iter(get_solver(name_solver, c) for c in target_algorithms_and_sections[name_solver])
    return iter[get_solver(name_solver, config_section)]
