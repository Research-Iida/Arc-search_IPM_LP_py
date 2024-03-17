import numpy as np

from ..logger import get_main_logger
from ..data_access import CsvHandler, MpsLoader
from ..problem import LPPreprocessor, LinearProgrammingProblemStandard as LPS
from ..solver import LPSolver
from ..solver.solver import LPVariables, SolvedDetail
from ..drawer import Drawer
from .decolators import deco_logging
from .define_paths import path_solved_result_by_problem, path_solved_result_by_solver_with_config

logger = get_main_logger()


def preprocess(
    problem_name: str, aMpsLoader: MpsLoader, aCsvHandler: CsvHandler
) -> LPS:
    """前処理を施し, 標準形となった Netlib LP を csv で書き込む
    """
    logger.info(f"Start loading problem '{problem_name}'")
    aLP_origin = aMpsLoader.run(problem_name).convert_standard()
    logger.info("End loading.")
    logger.info(f"Origin dimension: n: {aLP_origin.n}, m: {aLP_origin.m}")
    logger.info("Start preprocessing.")
    aLP = LPPreprocessor().run(aLP_origin)
    logger.info("End preprocessing.")
    logger.info("Start writing csv.")
    aCsvHandler.write_LP(aLP, problem_name)
    logger.info("End writing.")
    return aLP


def optimize(aLP: LPS, aLPSolver: LPSolver, v_0: LPVariables | None = None) -> SolvedDetail:
    """最適化の実行. ロギングなども同時に行う

    Args:
        aLP: 求解対象の線形計画問題
        aLPSolver: 線形計画問題の solver
        v_0: 初期点

    Returns:
        求解した結果
    """
    problem_name = aLP.name
    solver_name = aLPSolver.__class__.__name__

    # 入力をデコレータに渡すための実質のmain関数
    @deco_logging(problem_name, aLPSolver)
    def _optimize():
        return aLPSolver.run(aLP, v_0)

    output = _optimize()
    # 求解できなかったら warning
    if not output.aSolvedSummary.is_solved:
        msg = f"[{solver_name}] [{aLPSolver.config_section}] Algorithm cannot solve {problem_name}!"
        logger.warning(msg)
    return output


def solve(
    problem_name: str,
    aLPSolver: LPSolver,
    aMpsLoader: MpsLoader,
    aCsvHandler: CsvHandler
) -> SolvedDetail:
    """ベンチマークの問題を読み取り, 解く

    すでに問題を前処理したファイルが存在する場合, そこから読み取ることで時間を短縮する

    Args:
        problem_name: ベンチマーク問題の名前
        aLPSolver: 線形計画問題を解くためのソルバー. 抽象クラスなので, 実際に使用する際は
            ソルバーを指定

    Returns:
        SolvedDetail: 最適化によって作成された諸解群
    """
    # すでに前処理済みの問題であれば, csvファイルから読み込む
    if aCsvHandler.can_read_LP(problem_name):
        logger.info(f"There are preprocessed {problem_name} data.")
        logger.info(f"Read {problem_name} csv files.")
        aLP = aCsvHandler.read_LP(problem_name)
    # そうでなければ前処理を行い, csvファイルに書き込んでおく
    else:
        aLP = preprocess(problem_name, aMpsLoader, aCsvHandler)

    # 最適化
    return optimize(aLP, aLPSolver)


def solve_and_write(
    filename: str, solver: LPSolver, aMpsLoader: MpsLoader, aCsvHandler: CsvHandler,
    name_result: str, path_result: str
) -> SolvedDetail:
    """問題を解き, 結果を格納する. `__main__.py` で使用するので書き出しておく"""
    aSolvedDetail = solve(
        filename, solver,
        aMpsLoader=aMpsLoader, aCsvHandler=aCsvHandler
    )
    # 計算が終わるたびに都度書き込みを行う
    aCsvHandler.write_SolvedSummary(
        [aSolvedDetail.aSolvedSummary],
        name_result,
        path=path_result,
        is_append=True
    )
    return aSolvedDetail


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
