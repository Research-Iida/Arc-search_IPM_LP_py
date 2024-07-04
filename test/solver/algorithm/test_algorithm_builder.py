from src.solver.algorithm.algorithm_builder import AlgorithmBuilder
from src.solver.algorithm.inexact_interior_point_method import InexactArcSearchIPM
from src.solver.algorithm.initial_point_maker import LustingInitialPointMaker
from src.solver.algorithm.interior_point_method import ArcSearchIPM
from src.solver.algorithm.iterative_refinement import IterativeRefinementMethod
from src.solver.solved_checker import AbsoluteSolvedChecker, InexactSolvedChecker
from src.utils.config_utils import test_section


def test_Lusting_initial_point_maker():
    builder = AlgorithmBuilder("TEST_INITIAL_POINT_MAKER_LUSTING")
    test_obj = builder.build(algorithm="arc")
    assert isinstance(test_obj, ArcSearchIPM)
    assert isinstance(test_obj.initial_point_maker, LustingInitialPointMaker)


def test_solved_checker_absolute():
    builder = AlgorithmBuilder(test_section)
    test_obj = builder.build(algorithm="arc")
    assert isinstance(test_obj.solved_checker, AbsoluteSolvedChecker)


def test_build_inexact_arc_search_IPM():
    builder = AlgorithmBuilder(test_section)
    test_obj = builder.build(algorithm="inexact_arc")
    assert isinstance(test_obj, InexactArcSearchIPM)
    assert isinstance(test_obj.solved_checker, InexactSolvedChecker)


def test_build_iterative_refinement():
    builder = AlgorithmBuilder(test_section)
    test_obj = builder.build(algorithm="iterative_refinement")
    assert isinstance(test_obj, IterativeRefinementMethod)
    assert isinstance(test_obj.inner_algorithm, InexactArcSearchIPM)
