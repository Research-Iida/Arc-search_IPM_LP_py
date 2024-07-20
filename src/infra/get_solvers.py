from collections.abc import Iterator

from ..solver.solver import LPSolver
from ..utils.config_utils import default_section
from .algorithm_builder import AlgorithmBuilder

# 計算対象のアルゴリズム一覧
target_algorithms: list[str] = [
    # "line",
    # "arc",
    # "arc_restarting",
    # "arc_restarting_proven",
    "inexact_arc",
    "inexact_line",
    # "iterative_refinement",
]
# アルゴリズム別計算対象の config セクション一覧
# 計算対象にさせたくないアルゴリズムは, すべての config セクションをコメントアウトする
target_config_sections_by_algorithm: dict[str, list[str]] = {
    "line": [default_section],
    "arc": [default_section],
    "arc_restarting": [
        default_section,
        # "GUARANTEE_MAIN_RESIDUAL_DECREASING",
        # "IS_STOPPING_CRITERIA_ABSOLUTE",
        # "RESTART_01",
        # "RESTART_03",
        # "RESTART_07",
        # "RESTART_09",
        # "RESTART_001",
        # "RESTART_0001",
        # "RESTART_099",
        # "RESTART_0999",
        # "RESTART_1",
    ],
    "arc_restarting_proven": [
        default_section,
        # "RESTART_0",
        # "RESTART_0001",
        # "RESTART_0999",
        # "RESTART_1",
    ],
    "inexact_arc": [
        # "INEXACT_ARC_CG_MNES",
        # "INEXACT_ARC_CG_NES",
        # "INEXACT_ARC_CG_NES_LUSTING",
        "INEXACT_ARC_CG_NES_CONSTANT",
        # "INEXACT_ARC_EXACT_NES",
        # "INEXACT_ARC_EXACT_NES_LUSTING",
        # "INEXACT_ARC_EXACT_NES_CONSTANT",
        # "INEXACT_ARC_BICG_NES",
        # "INEXACT_ARC_BICGSTAB_NES",
        # "INEXACT_ARC_CGS_NES",
        # "INEXACT_ARC_QMR_NES",
        # "INEXACT_ARC_TFQMR_NES",
        # "INEXACT_ARC_HHLJULIA_NES",
        # "INEXACT_ARC_HHLJULIA_NES_IN_SERVER",
    ],
    "inexact_line": [
        # "INEXACT_LINE_CG_MNES",
        # "INEXACT_LINE_CG_NES",
        # "INEXACT_LINE_CG_NES_LUSTING",
        "INEXACT_LINE_CG_NES_CONSTANT",
        # "INEXACT_LINE_EXACT_NES",
        # "INEXACT_LINE_EXACT_NES_LUSTING",
        # "INEXACT_LINE_EXACT_NES_CONSTANT",
    ],
    "iterative_refinement": [
        # "INEXACT_ARC_CG_MNES",
        "INEXACT_ARC_CG_NES",
        "INEXACT_ARC_EXACT_NES",
        # "INEXACT_LINE_CG_MNES",
        "INEXACT_LINE_CG_NES",
        # "INEXACT_ARC_HHLJULIA_NES",
        # "INEXACT_ARC_HHLJULIA_NES_IN_SERVER",
    ],
}


class SolverSelectionError(Exception):
    """solver の選択に失敗したときに発生するエラー"""

    pass


def get_solver(algorithm: str, config_section: str) -> LPSolver:
    """線形計画問題のソルバー取得

    Args:
        solver: 取得したいソルバーの種類名
        config_section: 使用する config のセクション名
    """
    return LPSolver(AlgorithmBuilder(config_section).build(algorithm))


def get_solvers(name_solver: str | None, config_section: str | None) -> Iterator[LPSolver]:
    """対象のソルバー群から複数のソルバー取得"""
    if name_solver is None:
        if config_section is None:
            solvers = (get_solver(a, c) for a in target_algorithms for c in target_config_sections_by_algorithm[a])
        else:
            solvers = (get_solver(a, config_section) for a in target_algorithms)
    else:
        if config_section is None:
            solvers = (get_solver(name_solver, c) for c in target_config_sections_by_algorithm[name_solver])
        else:
            solvers = [get_solver(name_solver, config_section)]
    return iter(solvers)
