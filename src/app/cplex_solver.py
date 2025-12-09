# import time

import numpy as np
from docplex.mp.model import Model
from scipy.sparse import csr_matrix

from ..problem.problem import LinearProgrammingProblemStandard as LPS
from ..solver.solved_data import SolvedDetail
from ..solver.solver import ILPSolver
from ..solver.variables import LPVariables


class LPCPLEXSolver(ILPSolver):
    def _execute(self, problem: LPS, v_0: LPVariables | None) -> SolvedDetail:
        # start_time = time.perf_counter()

        # モデル作成
        mdl = Model(name=problem.name)
        # 内点法（バリア法）で解く
        mdl.parameters.lpmethod = 4
        # crossover を切る（反復回数の純粋性を保つ）
        mdl.parameters.barrier.crossover = -1
        # 許容誤差
        mdl.parameters.barrier.convergetol = 1e-9
        # 最大反復回数
        mdl.parameters.barrier.maxit = 100
        # 並列スレッド数
        mdl.parameters.threads = 1

        n = problem.n
        m = problem.m
        A: csr_matrix = problem.A
        b = problem.b
        c = problem.c

        # 変数 x >= 0
        x = mdl.continuous_var_list(n, lb=0.0, name="x")

        # 目的関数: min c^T x
        mdl.minimize(mdl.sum(c[j] * x[j] for j in range(n)))

        # 制約: A x = b （CSRを使って高速に追加）
        A_csr = A.tocsr()
        for i in range(m):
            start = A_csr.indptr[i]
            end = A_csr.indptr[i + 1]
            cols = A_csr.indices[start:end]
            vals = A_csr.data[start:end]

            mdl.add(mdl.sum(vals[k] * x[cols[k]] for k in range(len(cols))) == float(b[i]))

        # 求解
        sol = mdl.solve(log_output=True)
        # elapsed_time = time.perf_counter() - start_time

        # 実行不可能だった場合

        if sol is None:
            # TODO: 反復回数の反映
            # solved_summary = SolvedSummary(
            #     problem.name,
            #     self.solver_name,
            #     self.solver_config_section,
            #     False,
            #     problem.n,
            #     problem.m,
            #     False,
            #     elapsed_time=elapsed_time,
            # )
            return {
                "status": "infeasible_or_unbounded",
                "objective": None,
                "x": None,
            }

        x_val = np.array([sol.get_value(x[j]) for j in range(n)], dtype=float)

        return {
            "status": sol.solve_status,
            "objective": sol.objective_value,
            "x": x_val,
        }
