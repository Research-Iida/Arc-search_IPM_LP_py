"""LP の読み込み・書き込みに関する module"""

import abc

from ..logger import get_main_logger
from ..problem import LinearProgrammingProblem as LP
from ..problem import LinearProgrammingProblemStandard as LPS

logger = get_main_logger()


class CannotReadError(Exception):
    """この module で読み込みができない場合に発生するエラー"""

    pass


class ILPRepository(abc.ABC):
    """LPを読み込む際に必要になる処理についてまとめたインターフェース"""

    @abc.abstractmethod
    def get_problem_names(self) -> list[str]:
        """参照しているストレージに存在する `SIF` ファイルの一覧を取得"""
        pass

    @abc.abstractmethod
    def read_raw_LP(self, problem_name: str) -> LP:
        """MPS ファイルを読み込んで線形計画問題インスタンスを出力する

        Args:
            problem_name: 問題名. パスは含まない. `.SIF` はついていてもいなくてもよい

        Returns:
            LP: 線形計画問題のインスタンス
        """
        pass

    @abc.abstractmethod
    def write_LP(self, aLP: LPS, problem_name: str):
        """線形計画問題を csvファイルに書き出す

        前処理したものを書き出す前提のため, `processed` ディレクトリに書き出す
        """
        pass

    @abc.abstractmethod
    def can_read_processed_LP(self, problem_name: str) -> bool:
        """指定した問題がディレクトリに存在し, 読み取ることが可能か"""
        pass

    @abc.abstractmethod
    def read_processed_LP(self, problem_name: str) -> LPS:
        """線形計画問題に関する data を読み込み, 問題のクラスインスタンスを出力"""
        pass
