from ..logger import get_main_logger
from ..solver import LPSolver

logger = get_main_logger()


def deco_logging(problem_name: str, solver: LPSolver):
    """求解開始をログ, および slack に出力するデコレータ

    Args:
        problem_name: 求解対象の問題の名前
        solver: 求解する際の solver
    """
    solver_name = solver.__class__.__name__
    config_section = solver.config_section

    def _deco_logging(func):
        def wrapper(*args, **kwargs):
            msg_prefix = f"[{solver_name}] [{config_section}]"
            msg_start = f"{msg_prefix} Start solving {problem_name}."
            logger.info(msg_start)

            output = func(*args, **kwargs)

            msg_end = f"{msg_prefix} End solving {problem_name}."
            logger.info(msg_end)
            return output
        return wrapper
    return _deco_logging
