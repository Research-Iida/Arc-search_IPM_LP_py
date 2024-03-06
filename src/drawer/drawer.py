

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np

from ..utils import str_util
from ..solver.solver import SolvedDetail
from ..logger import get_main_logger

# logger の設定
logger = get_main_logger()


def deco_logging(doing: str):
    """実行開始・終了を logging するためのデコレータ"""
    def _deco_logging(func):
        def wrapper(*args, **kwargs):
            logger.info(f"Start {doing}...")
            func(*args, **kwargs)
            logger.info(f"Finish {doing}.")
        return wrapper
    return _deco_logging


class Drawer:
    """研究に関するグラフの描画をつかさどるクラス"""
    def __init__(
        self,
        path_output: str,
    ):
        """グラフの描画を行う上での初期設定
        """
        self.path_output = path_output

    @classmethod
    def add_suffix(cls, filename: str) -> str:
        """`.png` がファイル名についていなければ追加する"""
        suffix = ".png"
        return str_util.add_suffix(filename, suffix)

    @deco_logging("drawing optimal trajectory")
    def _draw_optimality_trajectories(
        self, aSolvedDetail: SolvedDetail,
        file_name: str = "optimal_trajectories"
    ):
        """"duality measure mu など, 最適解に近づいていれば0に収束する値の軌跡を描画する

        Args:
            aSolvedDetail: 最適解の詳細. 描画対象となるリストをそれぞれ格納している
            file_name: 描画するファイルの名前. `.png` を含まなくともよい
        """
        lst_mu = aSolvedDetail.lst_mu_by_iter
        lst_norm_main = aSolvedDetail.lst_max_norm_main_constraint_by_iter
        lst_norm_dual = aSolvedDetail.lst_max_norm_dual_constraint_by_iter

        # 描画内容が存在しなければ何もせず終了
        if len(lst_mu) == 0 and len(lst_norm_main) == 0 and len(lst_norm_dual) == 0:
            logger.info("No drawing data.")
            return

        fig, ax = plt.subplots()
        # 軸のスケールを対数に
        ax.set_yscale("log")
        # 軸ラベル
        ax.set_xlabel('iteration number')
        ax.plot(lst_mu, "-", label="$\mu$")
        ax.plot(lst_norm_main, "--", label="primal constraint max norm")
        ax.plot(lst_norm_dual, "-.", label="dual constraint max norm")
        ax.legend()
        ax.set_title("trajectories for optimal solution")

        fig.savefig(self.add_suffix(f"{self.path_output}{file_name}"))

    @deco_logging("drawing search step trajectory")
    def _draw_search_step_trajectories(
        self, aSolvedDetail: SolvedDetail,
        file_name: str = "search_step_trajectories"
    ):
        """"探索方向や step size がどのような軌跡を描いたのか確認する

        Args:
            aSolvedDetail: 最適解の詳細. 描画対象となるリストをそれぞれ格納している
            file_name: 描画するファイルの名前. `.png` を含まなくともよい
        """
        lst_alpha_main = aSolvedDetail.lst_main_step_size_by_iter
        lst_alpha_dual = aSolvedDetail.lst_dual_step_size_by_iter
        lst_norm_v_dot = aSolvedDetail.lst_norm_vdot_by_iter
        lst_norm_v_ddot = aSolvedDetail.lst_norm_vddot_by_iter

        # 描画内容が存在しなければ何もせず終了
        # TODO: 途中まで step 進んでても singular matirix になったせいで描画するものがなくなるのはおかしくね?
        # CRE-A で描画できなくなったので注意
        if len(lst_alpha_main) == 0 and len(lst_alpha_dual) == 0 and len(lst_norm_v_dot) == 0 and len(lst_norm_v_ddot) == 0:
            logger.info("No drawing data.")
            return

        fig, ax1 = plt.subplots()
        # グラフのグリッドをグラフの本体の下にずらす
        ax1.set_axisbelow(True)
        color_1 = cm.Set1.colors[0]

        # 2軸グラフの本体設定
        # もし main と dual で同じ step size であれば, 一つに統一
        if all(lst_alpha_main[i] == lst_alpha_dual[i] for i in range(len(lst_alpha_main))):
            ax1.plot(lst_alpha_main, "--", color=color_1, label="step size")
        else:
            ax1.plot(lst_alpha_main, "--", color=color_1, label="primal step size")
            ax1.plot(lst_alpha_dual, "-.", color=cm.Set1.colors[2], label="dual step size")

        # 軸の縦線の色を変更
        ax1.tick_params(axis='y', colors=color_1)
        # step size の幅は 0 ~ pi/2(line は1まで)なので, それよりちょっと大きいところまで目盛りをとる
        ax1.set_ylim(0, 1.6)
        ax1.set_xlabel("Iteration number")
        ax1.set_ylabel("alpha")
        # グラフの本体設定時に, ラベルを手動で設定する必要があるのは barplot のみ. plotは自動で設定される
        handler1, label1 = ax1.get_legend_handles_labels()

        # 探索方向のノルムを描画しない場合はそのまま出力
        if len(lst_norm_v_dot) == 0 and len(lst_norm_v_ddot) == 0:
            ax1.legend(
                handler1, label1,
                loc="lower left",
                borderaxespad=0.
            )
            plt.title("trajectories for step size")
            plt.savefig(self.add_suffix(f"{self.path_output}{file_name}"))
            return

        color_2 = cm.Set1.colors[1]
        ax2 = ax1.twinx()
        ax2.plot(lst_norm_v_dot, color=color_2, label="$\|\dot{v}\|$")
        ax2.plot(lst_norm_v_ddot, color=cm.Set1.colors[3], label="$\|\ddot{v}\|$")
        # axesオブジェクトに属するSpineオブジェクトの値を変更
        # 図を重ねてる関係で、ax2 のみいじる。
        ax2.spines['left'].set_color(color_1)
        ax2.spines['right'].set_color(color_2)
        ax2.tick_params(axis='y', colors=color_2)
        ax2.set_yscale("log")
        ax2.set_ylabel("search direction norm")
        handler2, label2 = ax2.get_legend_handles_labels()

        # 凡例をまとめて出力する
        ax1.legend(
            handler1 + handler2, label1 + label2,
            # loc="lower left",
            # borderaxespad=0.
        )
        ax1.set_title("trajectories for search direction and step size")
        fig.savefig(self.add_suffix(f"{self.path_output}{file_name}"))

    @deco_logging("drawing variable min values trajectory")
    def _draw_variables_min_values_trajectories(
        self, aSolvedDetail: SolvedDetail,
        file_name: str = "variable_min_value_trajectories"
    ):
        """"正の値になるべき x,s が本当に正の値のまま反復しているかを確認するための軌跡を描画

        Args:
            aSolvedDetail: 最適解の詳細. 描画対象となるリストをそれぞれ格納している
            file_name: 描画するファイルの名前. `.png` を含まなくともよい
        """
        lst_variables = aSolvedDetail.lst_variables_by_iter

        # 描画内容が存在しなければ何もせず終了
        if len(lst_variables) == 0:
            logger.info("No drawing data.")
            return

        # x,s のそれぞれの最小値を取得
        lst_min_x = [min(v.x) for v in lst_variables]
        lst_min_s = [min(v.s) for v in lst_variables]

        fig, ax = plt.subplots()

        # y軸を対数に
        ax.set_yscale("log")

        # 軸ラベル
        ax.set_xlabel('iteration number')
        ax.plot(lst_min_x, "-", label="min $x$")
        ax.plot(lst_min_s, "--", label="min $s$")
        ax.legend()
        ax.set_title("trajectories of variables min values")

        fig.savefig(self.add_suffix(f"{self.path_output}{file_name}"))

    @deco_logging("drawing variable max values trajectory")
    def _draw_variables_max_norms_trajectories(
        self, aSolvedDetail: SolvedDetail,
        file_name: str = "variable_max_value_trajectories"
    ):
        """"変数が発散しないまま反復しているかを確認するための軌跡を描画

        Args:
            aSolvedDetail: 最適解の詳細. 描画対象となるリストをそれぞれ格納している
            file_name: 描画するファイルの名前. `.png` を含まなくともよい
        """
        lst_variables = aSolvedDetail.lst_variables_by_iter

        # 描画内容が存在しなければ何もせず終了
        if len(lst_variables) == 0:
            logger.info("No drawing data.")
            return

        # それぞれの最大値を取得
        lst_max_x = [max(v.x) for v in lst_variables]
        lst_max_s = [max(v.s) for v in lst_variables]
        # y は負の値になりうるのでノルムを取る
        lst_max_ynorm = [np.linalg.norm(v.y, np.inf) for v in lst_variables]

        fig, ax = plt.subplots()

        # y軸を対数に
        ax.set_yscale("log")

        # 軸ラベル
        ax.set_xlabel('iteration number')
        ax.plot(lst_max_x, "-", label="max $x$")
        ax.plot(lst_max_ynorm, "-.", label="max $|y|$")
        ax.plot(lst_max_s, "--", label="max $s$")
        ax.legend()
        ax.set_title("trajectories of variables max values")

        fig.savefig(self.add_suffix(f"{self.path_output}{file_name}"))

    @deco_logging("drawing residuals inexact solution trajectory")
    def _draw_residuals_inexact_solution_trajectories(
        self, aSolvedDetail: SolvedDetail,
        file_name: str = "residuals_inexact_solution"
    ):
        """"inexact に解いた時の誤差を描画

        Args:
            aSolvedDetail: 最適解の詳細. 描画対象となるリストをそれぞれ格納している
            file_name: 描画するファイルの名前. `.png` を含まなくともよい
        """
        lst_residual_inexact_vdot = aSolvedDetail.lst_residual_inexact_vdot
        lst_residual_inexact_vddot = aSolvedDetail.lst_residual_inexact_vddot

        # 描画内容が存在しなければ何もせず終了
        if len(lst_residual_inexact_vdot) == 0 and len(lst_residual_inexact_vddot) == 0:
            logger.info("No drawing data.")
            return

        fig, ax = plt.subplots()

        # 軸のスケールを対数に
        ax.set_yscale("log")
        # 軸ラベル
        ax.set_xlabel('iteration number')

        color_1 = cm.Set1.colors[0]
        ax.plot(lst_residual_inexact_vdot, "-", color=color_1, label="$\dot{v}$ residual")
        if len(lst_residual_inexact_vddot) > 0:
            color_2 = cm.Set1.colors[1]
            ax.plot(lst_residual_inexact_vddot, "-", color=color_2, label="$\ddot{v}$ residual")

        lst_tolerance_inexact_vdot = aSolvedDetail.lst_tolerance_inexact_vdot
        lst_tolerance_inexact_vddot = aSolvedDetail.lst_tolerance_inexact_vddot
        if len(lst_tolerance_inexact_vddot) == 0 or np.all(np.array(lst_tolerance_inexact_vdot) == np.array(lst_tolerance_inexact_vddot)):
            ax.plot(lst_tolerance_inexact_vdot, "--", color=color_1, label="tolerance")
        else:
            ax.plot(lst_tolerance_inexact_vdot, "--", color=color_1, label="$\dot{v}$ tolerance")
            ax.plot(lst_tolerance_inexact_vddot, "--", color=color_2, label="$\ddot{v}$ tolerance")
        ax.legend()
        ax.set_title("trajectories for residuals of inexact solution")

        fig.savefig(self.add_suffix(f"{self.path_output}{file_name}"))

    def run(self, aSolvedDetail: SolvedDetail):
        """描画する対象をすべて実行する

        ファイル名はデフォルトのまま.
        '_draw_' で始まる関数をすべて実行する
        """
        for func_name in dir(self):
            if func_name.startswith("_draw_"):
                eval(f"self.{func_name}(data)", {}, {"self": self, "data": aSolvedDetail})

        plt.clf()
        plt.close()
