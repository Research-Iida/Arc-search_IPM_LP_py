import numpy as np

from ...logger import get_main_logger
from ...problem import LinearProgrammingProblemStandard as LPS
from ...solver.linear_system_solver.exact_linear_system_solver import AbstractLinearSystemSolver
from ...solver.search_direction_calculator.search_direction_calculator import AbstractSearchDirectionCalculator
from ...solver.variables import LPVariables
from ..python.mnes_search_direction_calculator import MNESSearchDirectionCalculator

logger = get_main_logger()


class JuliaMNESSearchDirectionCalculator(AbstractSearchDirectionCalculator):
    # 初期点時点で決定できるものや1回計算すればいいものは Attributes として使いまわす
    A_base_indexes: list[int] | None = None
    python_mnes_search_direction_calculator: MNESSearchDirectionCalculator

    def __init__(self, linear_system_solver: AbstractLinearSystemSolver):
        self.python_mnes_search_direction_calculator = MNESSearchDirectionCalculator(linear_system_solver)

    def select_base_indexes(self, problem: LPS) -> list[int]:
        pass

    def run(
        self,
        v: LPVariables,
        problem: LPS,
        right_hand_side: np.ndarray,
        tolerance: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """MNES の定式化で探索方向を解く. pure python の実装を使用"""
        return self.python_mnes_search_direction_calculator.run(v, problem, right_hand_side, tolerance)
