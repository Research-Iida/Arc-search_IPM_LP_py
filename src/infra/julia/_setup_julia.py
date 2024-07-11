def setup_julia():
    from julia import Pkg  # noqa: E402

    Pkg.activate(".")
