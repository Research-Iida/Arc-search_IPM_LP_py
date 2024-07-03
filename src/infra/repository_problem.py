"""LP の読み込み・書き込みに関する module
"""

import os
from pathlib import Path

import numpy as np
from pysmps import smps_loader as smps
from scipy.sparse import coo_matrix, lil_matrix, load_npz, save_npz

from ..logger import get_main_logger
from ..problem import LinearProgrammingProblem as LP
from ..problem import LinearProgrammingProblemStandard as LPS
from ..problem.repository import CannotReadError, ILPRepository
from ..utils import config_utils, str_util

logger = get_main_logger()


class LPRepository(ILPRepository):
    """LPを読み込む際に必要になる処理についてまとめたクラス

    SIF file は制約ごとに上限, 下限が設定されていたり, box constraint が存在したりする
    標準形式に起こす必要がある
    """

    def __init__(self, config_section: str = config_utils.default_section):
        """初期化. `data` ディレクトリへのパスを設定する"""
        config_ini = config_utils.read_config(section=config_section)

        self._path_netlib: Path = Path(config_ini.get("PATH_NETLIB"))
        self._path_data: Path = Path(config_ini.get("PATH_DATA"))
        self._path_processed: Path = Path(config_ini.get("PATH_PROCESSED"))
        self._path_result: Path = Path(config_ini.get("PATH_RESULT"))

    def get_problem_names(self) -> list[str]:
        """参照しているディレクトリに存在する `SIF` ファイルの一覧を取得

        問題名のみ取り出したいので, `.SIF` を削除して出力する
        """
        return [fullpath.name[:-4] for fullpath in self._path_netlib.glob("*.SIF")]

    def separate_by_constraint_type(self, types_constraint: list[str], A: lil_matrix, b_origin: np.ndarray):
        """制約の種類（等式, 上限, 下限）によって A,b のインスタンスを分ける

        Args:
            types_constraint: A, b の制約の種類. E or G or L
        """
        lst_index_eq = [idx for idx, val in enumerate(types_constraint) if val == "E"]
        A_E = A[lst_index_eq, :]
        b_E = b_origin[lst_index_eq]
        lst_index_ge = [idx for idx, val in enumerate(types_constraint) if val == "G"]
        A_G = A[lst_index_ge, :]
        b_G = b_origin[lst_index_ge]
        lst_index_le = [idx for idx, val in enumerate(types_constraint) if val == "L"]
        A_L = A[lst_index_le, :]
        b_L = b_origin[lst_index_le]
        return A_E, A_G, A_L, b_E, b_G, b_L

    def read_raw_LP(self, problem_name: str) -> LP:
        """MPS ファイルを読み込んで線形計画問題インスタンスを出力する

        Args:
            problem_name: 問題名. パスは含まない. `.SIF` はついていてもいなくてもよい

        Returns:
            LP: 線形計画問題のインスタンス
        """
        problem_filename = str_util.add_suffix(problem_name, ".SIF")
        mps_obj = smps.load_mps(self._path_netlib.joinpath(problem_filename))

        # 制約数
        m_origin = len(mps_obj[2])
        A_origin = coo_matrix(mps_obj[7]).tolil()

        # 変数次元数
        n_origin = len(mps_obj[3])
        c_origin = mps_obj[6]

        # すべて0であれば SIF file への記載が省略されるため, 明示的に設定
        if not len(c_origin):
            c_origin = np.zeros(n_origin)

        # 制約
        name_constraints = mps_obj[8]
        # 制約の右辺がすべて0の場合省略されるため, 明示的に設定
        if name_constraints:
            b_origin = mps_obj[9][name_constraints[0]]
        else:
            b_origin = np.zeros(m_origin)
        # 変数の上下限制約
        name_bounds = mps_obj[10]
        # 上下限制約がない場合記載が省略されるため, 明示的に設定
        if name_bounds:
            dct_bound = mps_obj[11][name_bounds[0]]
            lb_origin = dct_bound["LO"]
            ub_origin = dct_bound["UP"]
        else:
            lb_origin = np.zeros(n_origin)
            ub_origin = np.full(n_origin, np.inf)

        # 変数上限は制約に追加されるので, 上限があるものを取得しておく
        lst_index_lb = np.where(lb_origin != -np.inf)[0]
        lst_index_ub = np.where(ub_origin != np.inf)[0]
        lb = lb_origin[lst_index_lb]
        ub = ub_origin[lst_index_ub]

        # 等式制約, 上限 or 下限不等式制約に分ける
        types_constraint = mps_obj[5]
        A_E, A_G, A_L, b_E, b_G, b_L = self.separate_by_constraint_type(types_constraint, A_origin, b_origin)

        output = LP(A_E, b_E, A_G, b_G, A_L, b_L, lst_index_lb, lb, lst_index_ub, ub, c_origin, problem_name)
        return output

    def write_LP(self, aLP: LPS, problem_name: str):
        """線形計画問題を csvファイルに書き出す

        前処理したものを書き出す前提のため, `processed` ディレクトリに書き出す

        TODO:
            * 0は書き下すとファイルサイズが大きくなるので, 欠損させるようにしたい
        """
        # Aの書き出し, scipy.sparce の型なので別で書き出しする
        save_npz(self._path_processed.joinpath(f"{problem_name}_A.npz"), aLP.A)

        np.save(self._path_processed.joinpath(f"{problem_name}_b.npy"), aLP.b)
        np.save(self._path_processed.joinpath(f"{problem_name}_c.npy"), aLP.c)

        logger.info(f"'{problem_name}' is written in {self._path_processed}.")

    def can_read_processed_LP(self, problem_name: str) -> bool:
        """指定した問題がディレクトリに存在し, 読み取ることが可能か"""
        is_exist_A = os.path.exists(self._path_processed.joinpath(f"{problem_name}_A.npz"))
        is_exist_b = os.path.exists(self._path_processed.joinpath(f"{problem_name}_b.npy"))
        is_exist_c = os.path.exists(self._path_processed.joinpath(f"{problem_name}_c.npy"))
        return is_exist_A and is_exist_b and is_exist_c

    def read_processed_LP(self, problem_name: str) -> LPS:
        """線形計画問題に関するcsvファイルを読み込み, 問題のクラスインスタンスを出力

        csv上で欠損している箇所は0を代入する
        """
        # 読み込めない場合, エラーを返す
        if not self.can_read_processed_LP(problem_name):
            raise CannotReadError(f"{self._path_processed} 以下に前処理済みの '{problem_name}' data が存在しません.")

        A = load_npz(self._path_processed.joinpath(f"{problem_name}_A.npz"))

        b = np.load(self._path_processed.joinpath(f"{problem_name}_b.npy"))
        c = np.load(self._path_processed.joinpath(f"{problem_name}_c.npy"))

        return LPS(A, b, c, problem_name)
