import shutil

import numpy as np

from ..utils import config_utils
from ..logger import get_main_logger
from ..data_access import CsvHandler
from ..solver.solved_data import SolvedDetail
from ..drawer import Drawer
from .define_paths import path_solved_result_by_problem, path_solved_result_by_solver_with_config

logger = get_main_logger()


def copy_optimization_parameters(path_result: str, config_section: str = config_utils.default_section):
    """`config_optimizer.ini` を結果を格納するディレクトリにコピー

    Args:
        path_result (str): 結果を書き込む先のディレクトリ
    """
    config = config_utils.read_config(section=config_section)
    path_config = config.get("PATH_CONFIG")
    name_config_opt = config.get("CONFIG_OPTIMIZER")
    origin_config_opt = f"{path_config}{name_config_opt}"
    destination_config_opt = f"{path_result}{name_config_opt}"
    logger.info(f"Write {origin_config_opt} to {destination_config_opt}")
    shutil.copyfile(origin_config_opt, destination_config_opt)


def write_result_by_problem_solver_config(aSolvedDetail: SolvedDetail, path_result: str):
    """計算に関わるいろいろな設定を書き込む

    Args:
        path_result: result ディレクトリ. この下に `問題名/ソルバー名/セクション名` というディレクトリを作成して書き込みを行う
    """
    summary = aSolvedDetail.aSolvedSummary
    path_result_by_problem = path_solved_result_by_problem(path_result, summary.problem_name)
    path_result_by_problem_solver_config = path_solved_result_by_solver_with_config(path_result_by_problem, summary.solver_name, summary.config_section)

    # 変数の反復列をcsvで出力
    if len(aSolvedDetail.lst_variables_by_iter) > 0:
        variables = np.stack([np.concatenate([v.x, v.y, v.s]) for v in aSolvedDetail.lst_variables_by_iter])
        CsvHandler().write_numpy("variables", variables, path_result_by_problem_solver_config)

    # グラフ描画
    Drawer(path_result_by_problem_solver_config).run(aSolvedDetail)
