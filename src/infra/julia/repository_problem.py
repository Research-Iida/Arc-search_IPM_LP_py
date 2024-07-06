import numpy as np
from julia import Main
from scipy.sparse import coo_matrix

from src.problem.problem import LinearProgrammingProblem, LinearProgrammingProblemStandard

from ...problem.repository import ILPRepository
from ..python.repository_problem import LPRepository


class JuliaLPRepository(ILPRepository):
    """Julia を用いた repository"""

    pure_python_repository: LPRepository

    def __init__(self, config_section: str):
        """pure python の実装を持っておき, `read_raw_LP` 以外はそちらを使う"""
        Main.include("src/infra/julia/repository_problem.jl")

        self.pure_python_repository = LPRepository(config_section)

    def read_raw_LP(self, problem_name: str) -> LinearProgrammingProblem:
        """julia QPSReader を用いてファイルの読み込みを行う.
        TODO: 対応できるのが KEN-** のみ, 他も順次対応

        Args:
            problem_name (str): 問題名. 拡張子は不要（つけても問題なし）

        Returns:
            LinearProgrammingProblem: 線形計画問題
        """
        problem = Main.load_mps(problem_name)
        n = problem.n
        b_E: np.ndarray = problem.b_E
        A_E_coo = coo_matrix((problem.A_E_vals, (problem.A_E_rows, problem.A_E_cols)), shape=(b_E.shape[0], n))
        b_G: np.ndarray = problem.b_G
        A_G_coo = coo_matrix((problem.A_G_vals, (problem.A_G_rows, problem.A_G_cols)), shape=(b_G.shape[0], n))
        b_L: np.ndarray = problem.b_L
        A_L_coo = coo_matrix((problem.A_L_vals, (problem.A_L_rows, problem.A_L_cols)), shape=(b_L.shape[0], n))

        result = LinearProgrammingProblem(
            A_E=A_E_coo.tolil(),
            b_E=b_E,
            A_G=A_G_coo.tolil(),
            b_G=b_G,
            A_L=A_L_coo.tolil(),
            b_L=b_L,
            LB_index=problem.LB_index.tolist(),
            LB=problem.LB,
            UB_index=problem.UB_index.tolist(),
            UB=problem.UB,
            c=problem.c,
            name=problem_name,
        )
        return result

    def get_problem_names(self) -> list[str]:
        return self.pure_python_repository.get_problem_names()

    def write_LP(self, aLP: LinearProgrammingProblemStandard, problem_name: str):
        return self.pure_python_repository.write_LP(aLP, problem_name)

    def can_read_processed_LP(self, problem_name: str) -> bool:
        return self.pure_python_repository.can_read_processed_LP(problem_name)

    def read_processed_LP(self, problem_name: str) -> LinearProgrammingProblemStandard:
        return self.pure_python_repository.read_processed_LP(problem_name)
