from ..logger import get_main_logger
from ..solver.algorithm.algorithm import ILPSolvingAlgorithm
from ..solver.algorithm.inexact_arc_search_without_proof import InexactArcSearchIPMWithoutProof
from ..solver.algorithm.inexact_interior_point_method import InexactArcSearchIPM, InexactLineSearchIPM
from ..solver.algorithm.initial_point_maker import (
    ConstantInitialPointMaker,
    IInitialPointMaker,
    LustingInitialPointMaker,
    MehrotraInitialPointMaker,
    YangInitialPointMaker,
)
from ..solver.algorithm.interior_point_method import ArcSearchIPM, LineSearchIPM
from ..solver.algorithm.interior_point_method_with_restarting_strategy import (
    ArcSearchIPMWithRestartingStrategy,
    ArcSearchIPMWithRestartingStrategyProven,
)
from ..solver.algorithm.iterative_refinement import IterativeRefinementMethod
from ..solver.linear_system_solver import inexact_linear_system_solver
from ..solver.linear_system_solver.exact_linear_system_solver import (
    AbstractLinearSystemSolver,
    ExactLinearSystemSolver,
)
from ..solver.optimization_parameters import OptimizationParameters
from ..solver.search_direction_calculator.nes_search_direction_calculator import NESSearchDirectionCalculator
from ..solver.search_direction_calculator.search_direction_calculator import (
    AbstractSearchDirectionCalculator,
)
from ..solver.solved_checker import (
    AbsoluteSolvedChecker,
    ArcIPMWithRestartingProvenSolvedChecker,
    IterativeRefinementSolvedChecker,
    RelativeSolvedChecker,
    SolvedChecker,
)
from .python.mnes_search_direction_calculator import MNESSearchDirectionCalculator

logger = get_main_logger()


class SelectionError(Exception):
    """選択に失敗したときに発生するエラー"""

    pass


class AlgorithmBuilder:
    """アルゴリズムの構築に責務をもつ builder パターンクラス."""

    parameters: OptimizationParameters

    def __init__(self, config_section: str):
        # TODO: section はパラメータを使用したら大体決まるので削除したい
        self.config_section = config_section
        self.parameters = OptimizationParameters.import_(config_section)

    def get_solved_cheker(self) -> SolvedChecker:
        threshold_stop_criteria = self.parameters.STOP_CRITERIA_PARAMETER
        threshold_xs_negative = self.parameters.THRESHOLD_XS_NEGATIVE
        checker_type = self.parameters.SOLVED_CHECKER_TYPE

        match checker_type.lower():
            case "relative":
                return RelativeSolvedChecker(threshold_stop_criteria, threshold_xs_negative)
            case "absolute":
                return AbsoluteSolvedChecker(threshold_stop_criteria, threshold_xs_negative)
            case _:
                raise SelectionError(f"指定された収束判定が存在しません: {checker_type}")

    def get_initial_point_maker(self) -> IInitialPointMaker:
        """初期点を決定するクラスの取得

        Args:
            name (str | None): クラスに関連する名前. なければ Yang のものを出力

        Returns:
            IInitialPointMaker: 初期点決定クラス
        """
        name = self.parameters.INITIAL_POINT_MAKER
        match name.lower():
            case "yang":
                return YangInitialPointMaker(ExactLinearSystemSolver())
            case "mehrotra":
                return MehrotraInitialPointMaker(ExactLinearSystemSolver())
            case "lusting":
                return LustingInitialPointMaker(ExactLinearSystemSolver())
            case "yang_inexact":
                return YangInitialPointMaker(inexact_linear_system_solver.CGLinearSystemSolver())
            case "mehrotra_inexact":
                return MehrotraInitialPointMaker(inexact_linear_system_solver.CGLinearSystemSolver())
            case "lusting_inexact":
                return LustingInitialPointMaker(inexact_linear_system_solver.CGLinearSystemSolver())
            case "constant":
                return ConstantInitialPointMaker(self.parameters.INITIAL_POINT_SCALE)
            case _:
                raise SelectionError(f"指定された初期点決定法が存在しません: {name}")

    def get_linear_system_solver(self, str_solver: str) -> AbstractLinearSystemSolver:
        """内部で実行する linear system solver の取得"""
        match str_solver:
            case "CG":
                return inexact_linear_system_solver.CGLinearSystemSolver()
            case "BiCG":
                return inexact_linear_system_solver.BiCGLinearSystemSolver()
            case "BiCGStab":
                return inexact_linear_system_solver.BiCGStabLinearSystemSolver()
            case "CGS":
                return inexact_linear_system_solver.CGSLinearSystemSolver()
            case "QMR":
                return inexact_linear_system_solver.QMRLinearSystemSolver()
            case "TFQMR":
                return inexact_linear_system_solver.TFQMRLinearSystemSolver()
            case "HHLJulia":
                # いちいち import すると Julia のコンパイルに時間がかかるので指定されたときだけ
                from .julia.hhl import HHLJuliaLinearSystemSolver

                return HHLJuliaLinearSystemSolver(self.parameters.INEXACT_HHL_NUM_PHASE_ESTIMATOR_QUBITS)
            case "exact":
                return ExactLinearSystemSolver()
            case _:
                raise SelectionError(f"Don't match linear system solver: {str_solver}")

    def get_search_direction_calculator(
        self, str_calculator: str, linear_system_solver: AbstractLinearSystemSolver
    ) -> AbstractSearchDirectionCalculator:
        """内部で実行する linear system solver の取得"""
        match str_calculator:
            case "MNES":
                return MNESSearchDirectionCalculator(linear_system_solver)
            case "NES":
                return NESSearchDirectionCalculator(linear_system_solver)
            case _:
                raise SelectionError(f"Don't match search direction calculator: {str_calculator}")

    def get_inner_algorithm_for_iterative_refinement(
        self, initial_point_maker: IInitialPointMaker
    ) -> ILPSolvingAlgorithm:
        """Iterative Refinement 内部で実行する inexact solver の取得"""
        solved_checker = AbsoluteSolvedChecker(
            self.parameters.ITERATIVE_REFINEMENT_OPTIMAL_THRESHOLD_OF_SOLVER,
            self.parameters.THRESHOLD_XS_NEGATIVE,
        )
        linear_system_solver = self.get_linear_system_solver(self.parameters.INEXACT_LINEAR_SYSTEM_SOLVER)
        search_direction_calculator = self.get_search_direction_calculator(
            self.parameters.INEXACT_SEARCH_DIRECTION_CALCULATOR, linear_system_solver
        )

        str_algorithm = self.parameters.ITERATIVE_REFINEMENT_INNER_SOLVER
        match str_algorithm:
            case "inexact_arc":
                return InexactArcSearchIPM(
                    self.config_section,
                    self.parameters,
                    solved_checker,
                    initial_point_maker,
                    search_direction_calculator,
                )
            case "inexact_line":
                return InexactLineSearchIPM(
                    self.config_section,
                    self.parameters,
                    solved_checker,
                    initial_point_maker,
                    search_direction_calculator,
                )
            case "arc":
                return ArcSearchIPM(self.config_section, self.parameters, solved_checker, initial_point_maker)
            case "line":
                return LineSearchIPM(self.config_section, self.parameters, solved_checker, initial_point_maker)
            case _:
                raise SelectionError(f"Don't match inner algorithm in iterative refinement: {str_algorithm}")

    def build(self, algorithm: str) -> ILPSolvingAlgorithm:
        """線形計画問題の algorithm 取得

        Args:
            algorithm (str): 取得したい algorithm の種類名
        """
        initial_point_maker = self.get_initial_point_maker()

        match algorithm:
            case "arc":
                algorithm = ArcSearchIPM(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(),
                    initial_point_maker=initial_point_maker,
                )
            case "line":
                algorithm = LineSearchIPM(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(),
                    initial_point_maker=initial_point_maker,
                )
            case "arc_restarting":
                algorithm = ArcSearchIPMWithRestartingStrategy(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(),
                    initial_point_maker=initial_point_maker,
                )
            case "arc_restarting_proven":
                algorithm = ArcSearchIPMWithRestartingStrategyProven(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=ArcIPMWithRestartingProvenSolvedChecker(
                        self.parameters.STOP_CRITERIA_PARAMETER, self.parameters.THRESHOLD_XS_NEGATIVE
                    ),
                    initial_point_maker=initial_point_maker,
                )
            case "inexact_arc":
                linear_system_solver = self.get_linear_system_solver(self.parameters.INEXACT_LINEAR_SYSTEM_SOLVER)
                search_direction_calculator = self.get_search_direction_calculator(
                    self.parameters.INEXACT_SEARCH_DIRECTION_CALCULATOR, linear_system_solver
                )
                algorithm = InexactArcSearchIPM(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(),
                    initial_point_maker=initial_point_maker,
                    search_direction_calculator=search_direction_calculator,
                )
            case "inexact_line":
                linear_system_solver = self.get_linear_system_solver(self.parameters.INEXACT_LINEAR_SYSTEM_SOLVER)
                search_direction_calculator = self.get_search_direction_calculator(
                    self.parameters.INEXACT_SEARCH_DIRECTION_CALCULATOR, linear_system_solver
                )
                algorithm = InexactLineSearchIPM(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(),
                    initial_point_maker=initial_point_maker,
                    search_direction_calculator=search_direction_calculator,
                )
            case "iterative_refinement":
                threshold_stop_criteria = self.parameters.STOP_CRITERIA_PARAMETER
                threshold_xs_negative = self.parameters.THRESHOLD_XS_NEGATIVE
                algorithm = IterativeRefinementMethod(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=IterativeRefinementSolvedChecker(threshold_stop_criteria, threshold_xs_negative),
                    inner_algorithm=self.get_inner_algorithm_for_iterative_refinement(initial_point_maker),
                )
            case "inexact_arc_without_proof":
                linear_system_solver = self.get_linear_system_solver(self.parameters.INEXACT_LINEAR_SYSTEM_SOLVER)
                search_direction_calculator = self.get_search_direction_calculator(
                    self.parameters.INEXACT_SEARCH_DIRECTION_CALCULATOR, linear_system_solver
                )
                algorithm = InexactArcSearchIPMWithoutProof(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(),
                    initial_point_maker=initial_point_maker,
                    search_direction_calculator=search_direction_calculator,
                )
            case _:
                raise SelectionError(f"Don't match inner algorithm: {algorithm}")

        return algorithm
