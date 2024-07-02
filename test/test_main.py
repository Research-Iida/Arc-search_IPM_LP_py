import glob
import os

import pytest

from src.__main__ import decide_solved_problems, main, name_result
from src.problem.repository import LPRepository
from src.run_utils.define_paths import path_solved_result_by_date
from src.utils.config_utils import read_config, test_section

num_problem = 1
config = read_config(section=test_section)
aLPRepository = LPRepository(test_section)


def test_decide_solved_problems():
    test_lst = decide_solved_problems(aLPRepository, num_problem)

    assert len(test_lst) == num_problem
    assert "OSA-30" not in test_lst


@pytest.fixture(scope="module")
def remove_written_file():
    """前処理の結果が data/test/processed 配下に残るため, 不要な commit を行わないようテスト後に削除する.
    ここでエラーを起こすのであれば, main によって前処理が正しく行われなかったと判断できる"""
    path_processed = config.get("PATH_PROCESSED")
    file_list = glob.glob(f"{path_processed}KB2*")
    for filename in file_list:
        os.remove(filename)

    yield


def test_main(remove_written_file):
    """main関数のテスト. section は TEST なので書き込み先も result/test 配下
    問題ごとに軌跡の描画も行うため, 対象のディレクトリが存在するかも確認
    """
    # start_problem_number が 0 だと 25FV47 を解くことになり時間がかかるため, 1 の KB2 にする
    start_problem_number = 1
    main(
        num_problem=num_problem,
        name_solver="arc",
        config_section=test_section,
        start_problem_number=start_problem_number,
    )

    path_result = path_solved_result_by_date(config.get("PATH_RESULT"))
    assert os.path.exists(f"{path_result}{name_result}")
    target_problem_name = decide_solved_problems(aLPRepository, num_problem, start_problem_number)[0]
    assert os.path.exists(f"{path_result}{target_problem_name}/")
