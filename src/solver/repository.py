import abc
from pathlib import Path

from .solved_data import SolvedDetail, SolvedSummary


class ISolvedDataRepository(abc.ABC):
    @abc.abstractmethod
    def write_SolvedSummary(
        self, lst_solved_summary: list[SolvedSummary], name: str, path: Path, is_append: bool = False
    ):
        """最適化の実行によって得られた諸データを書き込む

        Args:
            path: 書き込み先
            is_append: 追記するか否か. 追記する場合は headerを抜く
        """
        pass

    @abc.abstractmethod
    def write_variables_by_iteration(self, aSolvedDetail: SolvedDetail, path: Path):
        """変数の反復列を出力"""
        pass
