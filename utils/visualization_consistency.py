import matplotlib.pyplot as plt
import seaborn as sns


def set_plot_style():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"""
            \usepackage[utf8]{inputenc}
            \usepackage[T1]{fontenc}
            \usepackage[polish]{babel}
            \usepackage{amsmath}
        """,
    })
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        font_scale=1.2
    )