from ..optimization_parameters import OptimizationParameters
from ..solved_checker import (
    AbsoluteSolvedChecker,
    InexactSolvedChecker,
    IterativeRefinementSolvedChecker,
    RelativeSolvedChecker,
    SolvedChecker,
)
from .algorithm import ILPSolvingAlgoritm
from .inexact_interior_point_method import InexactArcSearchIPM, InexactLineSearchIPM
from .initial_point_maker import (
    ConstantInitialPointMaker,
    IInitialPointMaker,
    LustingInitialPointMaker,
    MehrotraInitialPointMaker,
    YangInitialPointMaker,
)
from .interior_point_method import ArcSearchIPM, LineSearchIPM
from .interior_point_method_with_restarting_strategy import (
    ArcSearchIPMWithRestartingStrategy,
    ArcSearchIPMWithRestartingStrategyProven,
)
from .iterative_refinement import IterativeRefinementMethod


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

    def get_solved_cheker(self, algorithm: str) -> SolvedChecker:
        threshold_stop_criteria = self.parameters.STOP_CRITERIA_PARAMETER
        threshold_xs_negative = self.parameters.THRESHOLD_XS_NEGATIVE
        is_stop_relative = self.parameters.IS_STOPPING_CRITERIA_RELATIVE

        if algorithm == "iterative_refinement":
            return IterativeRefinementSolvedChecker(threshold_stop_criteria, threshold_xs_negative)

        if algorithm in {"inexact_arc", "inexact_line"}:
            return InexactSolvedChecker(threshold_stop_criteria, threshold_xs_negative, is_stop_relative)

        if is_stop_relative:
            return RelativeSolvedChecker(threshold_stop_criteria, threshold_xs_negative)

        return AbsoluteSolvedChecker(threshold_stop_criteria, threshold_xs_negative)

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
                result = YangInitialPointMaker()
            case "mehrotra":
                result = MehrotraInitialPointMaker()
            case "lusting":
                result = LustingInitialPointMaker()
            case "constant":
                result = ConstantInitialPointMaker(self.parameters.INITIAL_POINT_SCALE)
            case _:
                raise SelectionError(f"指定された初期点決定法が存在しません: {name}")

        return result

    def build(self, algorithm: str) -> ILPSolvingAlgoritm:
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
                    solved_checker=self.get_solved_cheker(algorithm),
                    initial_point_maker=initial_point_maker,
                )
            case "line":
                algorithm = LineSearchIPM(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(algorithm),
                    initial_point_maker=initial_point_maker,
                )
            case "arc_restarting":
                algorithm = ArcSearchIPMWithRestartingStrategy(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(algorithm),
                    initial_point_maker=initial_point_maker,
                )
            case "arc_restarting_proven":
                algorithm = ArcSearchIPMWithRestartingStrategyProven(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(algorithm),
                    initial_point_maker=initial_point_maker,
                )
            case "inexact_arc":
                algorithm = InexactArcSearchIPM(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(algorithm),
                    initial_point_maker=initial_point_maker,
                )
            case "inexact_line":
                algorithm = InexactLineSearchIPM(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(algorithm),
                    initial_point_maker=initial_point_maker,
                )
            case "iterative_refinement":
                algorithm = IterativeRefinementMethod(
                    self.config_section,
                    parameters=self.parameters,
                    solved_checker=self.get_solved_cheker(algorithm),
                    initial_point_maker=initial_point_maker,
                )
            case _:
                raise SelectionError(f"指定されたアルゴリズムが存在しません: {algorithm}")

        return algorithm
