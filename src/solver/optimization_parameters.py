from __future__ import annotations

import pydantic

from ..utils import config_utils


@pydantic.dataclasses.dataclass
class OptimizationParameters:
    STOP_CRITERIA_PARAMETER: float
    CALC_TIME_UPPER: int
    ITER_UPPER: int
    ITER_UPPER_COEF: int
    MIN_STEP_SIZE: float
    THRESHOLD_XS_NEGATIVE: float
    IS_STOPPING_CRITERIA_RELATIVE: bool
    INITIAL_POINT_SCALE: int

    IPM_COEF_GUARANTEEING_XS_POSITIVENESS: float
    IPM_LOWER_BOUND_OF_X_TRUNCATION: float

    RESTART_COEF_RESTARTING: float
    RESTART_IS_GUARANTEEING_MAIN_RESIDUAL_DECREASING: bool
    RESTART_COEF_CENTER_PATH_NEIGHBORHOOD: float

    INEXACT_LINEAR_SYSTEM_SOLVER: str
    INEXACT_SEARCH_DIRECTION_CALCULATOR: str
    INEXACT_COEF_OF_ARMIJO_RULE: float
    INEXACT_CENTERING_PARAMETER: float
    INEXACT_TOLERANCE_OF_RESIDUAL_OF_LINEAR_SYSTEM: float
    INEXACT_COEF_NEIGHBORHOOD_DUALITY: float
    INEXACT_COEF_NEIGHBORHOOD_CONSTRAINTS: float
    INEXACT_SOLVE_EXACTLY_FROM_THE_MIDDLE: bool
    INEXACT_HHL_NUM_PHASE_ESTIMATOR_QUBITS: int

    ITERATIVE_REFINEMENT_INNER_SOLVER: str
    ITERATIVE_REFINEMENT_OPTIMAL_THRESHOLD_OF_SOLVER: float
    ITERATIVE_REFINEMENT_SCALING_MULTIPLIER: float
    ITERATIVE_REFINEMENT_ITER_UPPER: int

    # TODO: クラス自体が読み込みに責務を持つのはおかしい. repository があるべき
    @classmethod
    def import_(cls, config_section: str) -> OptimizationParameters:
        config = config_utils.read_config(section=config_section)
        filename_config_opt = config.get("PATH_CONFIG") + config.get("CONFIG_OPTIMIZER")
        config_opt = config_utils.read_config(filename_config_opt, section=config_section)
        return cls(**config_opt)
