import time
from datetime import datetime

from docplex.mp.model import Model
from scipy.sparse import csr_matrix

from ..problem.problem import LinearProgrammingProblemStandard as LPS
from ..solver.optimization_parameters import OptimizationParameters
from ..solver.solved_data import SolvedDetail
from ..solver.solver import ILPSolver
from ..solver.variables import LPVariables


class LPCPLEXSolver(ILPSolver):
    config_section: str
    parameters: OptimizationParameters

    def __init__(
        self,
        config_section: str,
        parameters: OptimizationParameters,
    ):
        self.config_section = config_section
        self.parameters = parameters

    @property
    def solver_name(self) -> str:
        return "CPLEX"

    @property
    def solver_config_section(self) -> str:
        return self.config_section

    def _execute(self, problem: LPS, v_0: LPVariables | None) -> SolvedDetail:
        start_time = time.perf_counter()

        # モデル作成, 設定
        mdl = Model(name=problem.name)
        # 内点法（バリア法）で解く
        mdl.parameters.lpmethod = 4
        # crossover を切る（反復回数の純粋性を保つ）
        mdl.parameters.barrier.crossover = 1
        # 許容誤差
        mdl.parameters.barrier.convergetol = 1e-9
        # 並列スレッド数
        mdl.parameters.threads = 1
        # duality measure 閾値
        mdl.parameters.barrier.convergetol = self.parameters.STOP_CRITERIA_PARAMETER
        # 時間制限（秒）
        mdl.parameters.timelimit = self.parameters.CALC_TIME_UPPER

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
        with open(f"log/{datetime.now().strftime('%Y%m%d%H%M%S')}_{problem.name}_cplex.log", "w") as f:
            mdl.log_output = f
            sol = mdl.solve()
        elapsed_time = time.perf_counter() - start_time

        # 実行不可能だった場合
        if sol is None:
            solved_summary = self.create_solved_summary(
                problem=problem,
                is_error=False,
                is_solved=False,
                elapsed_time=elapsed_time,
            )
            solved_detail = SolvedDetail(solved_summary, v_0, problem, v_0, problem)
            return solved_detail

        # 情報取得
        cpx = mdl.get_cplex()
        barrier_iter_num = cpx.solution.progress.get_num_barrier_iterations()

        # TODO: LPVariables の中身を格納できるようにする
        # x_val = np.array([sol.get_value(x[j]) for j in range(n)], dtype=float)
        solved_summary = self.create_solved_summary(
            problem=problem,
            is_error=False,
            is_solved=True,
            iter_num=barrier_iter_num,
            elapsed_time=elapsed_time,
            obj=sol.objective_value,
        )
        solved_detail = SolvedDetail(solved_summary, v_0, problem, v_0, problem)
        return solved_detail
