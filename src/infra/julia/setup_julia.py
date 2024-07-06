from julia import Julia


def setup_julia():
    # サーバー環境で実行するためのおまじない
    Julia(compiled_modules=False)
    from julia import Pkg  # noqa: E402

    Pkg.activate(".")
