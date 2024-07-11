"""Python と julia を両立して使用する際に使用するコード
おまじないが必要になるため, julia を使用するパッケージはかならずこの `__init__.py` から読み込む
"""
from julia import Julia

# サーバー環境で実行するためのおまじない
Julia(compiled_modules=False)

from ._setup_julia import setup_julia
from .repository_problem import JuliaLPRepository
