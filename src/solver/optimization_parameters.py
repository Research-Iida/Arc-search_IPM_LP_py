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

    @classmethod
    def import_(cls, config_section: str) -> 'OptimizationParameters':
        config = config_utils.read_config(section=config_section)
        filename_config_opt = config.get("PATH_CONFIG") + config.get("CONFIG_OPTIMIZER")
        config_opt = config_utils.read_config(filename_config_opt, section=config_section)
        return cls(**config_opt)
