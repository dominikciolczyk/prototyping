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

from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import pandas as pd

# Spójne parametry linii, jak w EDA
_DEFAULT_LINE = dict(linewidth=1, marker="o", markersize=3, linestyle="--")

def set_common_date_axis(ax: plt.Axes, interval: int = 4) -> None:
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: mdates.num2date(x).strftime("%Y-%m-%d")))

def format_date_axis(fig: plt.Figure, rotation: int = 0) -> None:
    fig.autofmt_xdate(rotation=rotation)

def save_pdf(fig: plt.Figure, out_path: Path) -> str:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)
    return str(out_path.with_suffix(".pdf"))

def plot_training_curves(
        train_losses: List[float],
        val_losses: Optional[List[float]],
        out_path: Path,
) -> str:
    """Wykres przebiegu uczenia w spójnym stylu."""
    set_plot_style()
    fig, ax = plt.subplots(figsize=(6, 3))
    epochs = list(range(len(train_losses)))
    sns.lineplot(x=epochs, y=train_losses, ax=ax, label="zbiór treningowy", **_DEFAULT_LINE)
    if val_losses:
        sns.lineplot(x=epochs, y=val_losses, ax=ax, label="zbiór walidacyjny", **_DEFAULT_LINE)
    ax.set_xlabel("epoka treningu")
    ax.set_ylabel("wartość funkcji straty")
    ax.legend()
    return save_pdf(fig, Path(out_path))

def plot_forecasts(
        merged: Dict[str, pd.DataFrame],
        selected_columns: List[str],
        out_dir: Path,
        col_label_map: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Rysuje prognozy (rzeczywiste, model, baseline) w stylu jak w EDA.
    Oczekuje kolumn: <col>, <col>_pred_model, <col>_pred_baseline.
    """
    set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    for vm_id, df in merged.items():
        n = max(1, len(selected_columns))
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.8 * n), sharex=True)
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, selected_columns):
            base_label = col_label_map.get(col, col) if col_label_map else col

            # rzeczywiste
            sns.lineplot(x=df.index, y=df[col], ax=ax, label="rzeczywiste", **_DEFAULT_LINE)

            # model / baseline – jeśli są
            m_col = f"{col}_pred_model"
            b_col = f"{col}_pred_baseline"
            if m_col in df:
                sns.lineplot(x=df.index, y=df[m_col], ax=ax, label="CNN-LSTM", **_DEFAULT_LINE)
            if b_col in df:
                sns.lineplot(x=df.index, y=df[b_col], ax=ax, label="model bazowy (max)", **_DEFAULT_LINE)

            ax.set_xlabel("czas pomiaru (interwał dwugodzinny)")
            ax.set_ylabel(base_label)
            set_common_date_axis(ax, interval=4)

        format_date_axis(fig, rotation=0)
        axes[-1].legend()
        paths[vm_id] = save_pdf(fig, out_dir / f"{vm_id}_forecast")

    return paths