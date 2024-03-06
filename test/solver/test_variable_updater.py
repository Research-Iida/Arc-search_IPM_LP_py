import numpy as np

from src.utils import config_utils
from src.solver.variable_updater import ArcVariableUpdater

config_section = "TEST"


def test_max_alpha_guarantee_positive_with_line():
    """一次微分の値が0より大きい値の中で x / x_dot の中の最小値が出力されることを確認"""
    x = np.array([1, 2, 3, 1])
    x_dot = np.array([2, 0, 1, -1])
    alpha = ArcVariableUpdater(0).max_alpha_guarantee_positive_with_line(x, x_dot)
    assert alpha == 1 / 2


def test_update_variable():
    """次の点が正しく出力されているか確認"""
    x = np.array([1, 1])
    x_dot = np.array([2, 1])
    x_ddot = np.array([-1, -3])
    alpha_x = 0.7
    test_vec = ArcVariableUpdater(0).run(x, x_dot, x_ddot, alpha_x)
    sol_vec = x - x_dot * np.sin(alpha_x) + x_ddot * (1 - np.cos(alpha_x))
    np.testing.assert_array_equal(test_vec, sol_vec)


def test_max_step_size_guarantee_positive():
    """step size の最大値が各ケースごとに正しく出力されているか"""
    config_base = config_utils.read_config(section=config_section)
    config_opt = config_utils.read_config(
        config_base.get("PATH_CONFIG") + config_base.get("CONFIG_OPTIMIZER"),
        section=config_section
    )
    delta_xs = config_opt.getfloat("IPM_COEF_GUARANTEEING_XS_POSITIVENESS")
    # 本テストケースでよく使用する変数・関数の設定
    x = np.array([1])
    max_alpha = np.pi / 2
    x_minus_delta = x * (1 - delta_xs)

    def calc_alpha(x_dot: np.array, x_ddot: np.array):
        output = ArcVariableUpdater(delta_xs).max_step_size_guarantee_positive(
            x, x_dot, x_ddot
        )
        return output

    # Case 1
    x_dot = np.array([0])
    x_ddot = np.array([-2])
    test_sol = np.arccos((x_minus_delta + x_ddot) / x_ddot)
    assert calc_alpha(x_dot, x_ddot) == test_sol
    x_ddot = np.array([-0.5])
    assert calc_alpha(x_dot, x_ddot) == max_alpha

    # Case 2
    x_dot = np.array([2])
    x_ddot = np.array([0])
    assert calc_alpha(x_dot, x_ddot) == np.arcsin(x_minus_delta / x_dot)
    x_dot = np.array([0.5])
    assert calc_alpha(x_dot, x_ddot) == max_alpha

    # Case 3
    x_dot = np.array([2])
    x_ddot = np.array([1])
    denom = np.sqrt(x_dot**2 + x_ddot**2)
    first_term = np.arcsin((x_minus_delta + x_ddot) / denom)
    test_sol = first_term - np.arcsin(x_ddot / denom)
    assert calc_alpha(x_dot, x_ddot) == test_sol
    x_dot = np.array([0.5])
    assert calc_alpha(x_dot, x_ddot) == max_alpha

    # Case 4
    x_dot = np.array([1])
    x_ddot = np.array([-1])
    denom = np.sqrt(x_dot**2 + x_ddot**2)
    first_term = np.arcsin((x_minus_delta + x_ddot) / denom)
    test_sol = first_term + np.arcsin(-x_ddot / denom)
    assert calc_alpha(x_dot, x_ddot) == test_sol
    x_dot = np.array([0.1])
    x_ddot = np.array([-0.1])
    assert calc_alpha(x_dot, x_ddot) == max_alpha

    # Case 5
    x_dot = np.array([-1])
    x_ddot = np.array([-2])
    denom = np.sqrt(x_dot**2 + x_ddot**2)
    first_term = -np.arcsin(-(x_minus_delta + x_ddot) / denom)
    alpha = first_term - np.arcsin(-x_ddot / denom)
    assert calc_alpha(x_dot, x_ddot) == np.pi + alpha
    x_ddot = np.array([-1.5])
    assert calc_alpha(x_dot, x_ddot) == max_alpha

    # Case 6
    x_dot = np.array([-1])
    x_ddot = np.array([1])
    assert calc_alpha(x_dot, x_ddot) == max_alpha

    # Case 7
    x_dot = np.array([0])
    x_ddot = np.array([0])
    assert calc_alpha(x_dot, x_ddot) == max_alpha
