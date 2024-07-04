from pathlib import Path

from ..logger import get_main_logger
from ..problem import LinearProgrammingProblemStandard as LPS
from ..problem import LPPreprocessor
from ..problem.repository import ILPRepository
from ..slack.slack import get_slack_api
from .repository import ISolvedDataRepository
from .solved_data import SolvedDetail
from .solver import LPSolver
from .variables import LPVariables

logger = get_main_logger()
aSlack = get_slack_api()


def preprocess(problem_name: str, aLPRepository: ILPRepository) -> LPS:
    """前処理を施し, 標準形となった Netlib LP を csv で書き込む"""
    logger.info(f"Start loading problem '{problem_name}'")
    aLP_origin = aLPRepository.read_raw_LP(problem_name).convert_standard()
    logger.info("End loading.")
    logger.info(f"Origin dimension: n: {aLP_origin.n}, m: {aLP_origin.m}")
    logger.info("Start preprocessing.")
    aLP = LPPreprocessor().run(aLP_origin)
    logger.info("End preprocessing.")
    logger.info("Start writing data.")
    aLPRepository.write_LP(aLP, problem_name)
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

    msg_prefix = f"[{aLPSolver.algorithm.__class__.__name__}] [{aLPSolver.algorithm_config_section}]"
    msg_start = f"{msg_prefix} Start solving {problem_name}."
    logger.info(msg_start)
    aSlack.notify(msg_start)

    output = aLPSolver.run(aLP, v_0)

    msg_end = f"{msg_prefix} End solving {problem_name}."
    logger.info(msg_end)
    aSlack.notify(msg_end)

    # 求解できなかったら warning
    if not output.aSolvedSummary.is_solved:
        msg = f"{msg_prefix} Algorithm cannot solve {problem_name}!"
        logger.warning(msg)
        aSlack.notify(msg)
    return output


def solve(problem_name: str, aLPSolver: LPSolver, aLPRepository: ILPRepository) -> SolvedDetail:
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
    if aLPRepository.can_read_processed_LP(problem_name):
        logger.info(f"There are preprocessed {problem_name} data.")
        logger.info(f"Read {problem_name} data.")
        aLP = aLPRepository.read_processed_LP(problem_name)
    # そうでなければ前処理を行い, csvファイルに書き込んでおく
    else:
        aLP = preprocess(problem_name, aLPRepository)

    # 最適化
    return optimize(aLP, aLPSolver)


def solve_and_write(
    filename: str,
    solver: LPSolver,
    aLPRepository: ILPRepository,
    aSolvedDataRepository: ISolvedDataRepository,
    name_result: str,
    path_result: str,
) -> SolvedDetail:
    """問題を解き, 結果を格納する. `__main__.py` で使用するので書き出しておく"""
    aSolvedDetail = solve(filename, solver, aLPRepository=aLPRepository)
    # 計算が終わるたびに都度書き込みを行う
    aSolvedDataRepository.write_SolvedSummary(
        [aSolvedDetail.aSolvedSummary], name_result, path=Path(path_result), is_append=True
    )
    return aSolvedDetail
