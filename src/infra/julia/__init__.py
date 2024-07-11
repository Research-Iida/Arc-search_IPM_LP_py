"""Python と julia を両立して使用する際に使用するコード
仮想環境の activate とおまじないが必要なので `__init__.py` に記載
"""

# from julia import Julia

# # サーバー環境で実行するためのおまじない
# Julia(compiled_modules=False)

from julia import Pkg  # noqa: E402

Pkg.activate(".")
