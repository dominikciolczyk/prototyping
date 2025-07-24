from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from torch.fx.experimental.proxy_tensor import decompose
from zenml.logger import get_logger

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose

logger = get_logger(__name__)

DATE_FORMAT = "%d-%m-%Y"

COLUMNS_TO_SUMMARIZE = [
    "CPU_USAGE_MHZ",
    "MEMORY_USAGE_KB",
    "NODE_1_DISK_IO_RATE_KBPS",
    "NODE_1_NETWORK_TR_KBPS",
]

COLUMNS_TO_YLABEL = {
    "CPU_USAGE_MHZ": "wykorzystanie CPU",
    "MEMORY_USAGE_KB": "zużycie pamięci",
    "NODE_1_DISK_IO_RATE_KBPS": "operacje dyskowe",
    "NODE_1_NETWORK_TR_KBPS": "transfer sieciowy",
}

COLUMNS_TO_YUNITS = {
    "CPU_USAGE_MHZ": "[MHz]",
    "MEMORY_USAGE_KB": "[KB]",
    "NODE_1_DISK_IO_RATE_KBPS": "[KB/s]",
    "NODE_1_NETWORK_TR_KBPS": "[KB/s]",
}

def get_column_fullname(col: str) -> str:
    return f"{COLUMNS_TO_YLABEL[col]} {COLUMNS_TO_YUNITS[col]}"

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

set_plot_style()

def set_common_date_axis(ax: plt.Axes, interval: int = 4) -> None:
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))

    def format_func(x, _):
        dt = mdates.num2date(x)
        return dt.strftime('%Y-%m-%d')

    ax.xaxis.set_major_formatter(FuncFormatter(format_func))

def format_date_axis(fig: plt.Figure, rotation: int = 30) -> None:
    fig.autofmt_xdate(rotation=rotation)

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".pdf"), format="pdf")
    plt.close(fig)

def _plot_time_series(df: pd.DataFrame, out_path: Path) -> None:
    for col in COLUMNS_TO_YLABEL:
        fig, ax = plt.subplots(figsize=(9, 3))

        sns.lineplot(x=df.index, y=df[col], ax=ax, linewidth=1)

        ax.set_xlabel("czas pomiaru (interwał dwugodzinny)")
        ax.set_ylabel(get_column_fullname(col))

        set_common_date_axis(ax, interval=4)
        format_date_axis(fig, rotation=30)

        _save_fig(fig, out_path.with_name(f"{out_path.stem}_{col}"))

def _plot_histograms(df: pd.DataFrame, out_dir: Path) -> None:
    for col in COLUMNS_TO_YLABEL:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df[col].dropna(), ax=ax, bins=100, kde=False)

        ax.set_xscale("log")

        ax.set_xlabel(f"{col} (skala logarytmiczna)")
        ax.set_xlabel(get_column_fullname(col))
        _save_fig(fig, out_dir / f"hist_{col}")

def _plot_distributions(df: pd.DataFrame, out_dir: Path) -> None:
    for col in COLUMNS_TO_YLABEL:
        fig, ax = plt.subplots(figsize=(4, 3))

        try:
            sns.kdeplot(df[col].dropna(), ax=ax, fill=True, bw_adjust=0.5)
        except Exception as e:
            logger.warning(f"Could not plot KDE for {col}: {e}")
            continue

        ax.set_xlabel(get_column_fullname(col))
        ax.set_ylabel("gęstość")

        _save_fig(fig, out_dir / f"density_{col}")

def _plot_correlation(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    df = df.copy()
    df = df[COLUMNS_TO_YLABEL.keys()]
    df.columns = [COLUMNS_TO_YLABEL[col] for col in df.columns]
    corr = df.corr()
    sns.heatmap(
        corr,
        annot=True,
        cmap="viridis",
        ax=ax,
        cbar_kws={'shrink': 0.6},
        square=True,
        xticklabels=True,
        yticklabels=True,
        fmt=".2f"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout()
    _save_fig(fig, out_path)

def _plot_acf_pacf(df: pd.DataFrame, column: str, output_path: Path, lags: int = 72):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    plot_acf(df[column].dropna(), lags=lags, ax=axes[0])
    plot_pacf(df[column].dropna(), lags=lags, ax=axes[1])
    axes[0].set_title("autokorelacja")
    axes[1].set_title("częściowa autokorelacja")
    fig.tight_layout()
    fig.savefig(output_path / f"{column}_acf_pacf.pdf")
    plt.close(fig)

def _plot_seasonal_decompositions(df: pd.DataFrame, column: str, output_path: Path, periods: list):
    df = df.copy()
    df = df[COLUMNS_TO_YLABEL.keys()]
    df.columns = [get_column_fullname(col) for col in df.columns]
    column = get_column_fullname(column)
    for period in periods:
        decomposition = seasonal_decompose(df[column].dropna(), period=period, model='additive')
        fig = decomposition.plot()
        fig.set_size_inches(10, 6)
        fig.tight_layout()
        fig.savefig(output_path / f"{column}_decomposition_period_{period}.pdf")
        plt.close(fig)

def data_summarizer(
    dfs: Dict[str, pd.DataFrame],
    output_dir: str = "report_output"
) -> None:
    _ensure_dir(Path(output_dir))

    summary_rows = []
    summary_by_year = {2020: [], 2022: []}

    for vm_name, df in dfs.items():
        if df.empty:
            raise ValueError(f"DataFrame for VM '{vm_name}' is empty.")
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        if not (vm_name.startswith("2020") or vm_name.startswith("2022")):
            raise ValueError(f"VM name '{vm_name}' must start with 2020 or 2022.")
        year = 2022 if vm_name.startswith("2022") else 2020

        summary_by_year[year].append({
            "VM": vm_name,
            "count": len(df),
            "start": df.index.min().strftime(DATE_FORMAT),
            "end": df.index.max().strftime(DATE_FORMAT),
        })

        stats = {"VM": vm_name, "count": int(df.shape[0]), "start": df.index.min().strftime("%Y-%m-%d"), "end": df.index.max().strftime("%Y-%m-%d")}
        for col in df.columns:
            if col not in COLUMNS_TO_SUMMARIZE:
                continue
            desc = df[col].describe()
            stats.update({
                f"{col}_mean": desc["mean"],
                f"{col}_median": desc["50%"],
                f"{col}_std": desc["std"],
                f"{col}_min": desc["min"],
                f"{col}_q25": desc["25%"],
                f"{col}_q75": desc["75%"],
                f"{col}_max": desc["max"],
            })
        summary_rows.append(stats)

    df_stats = pd.DataFrame(summary_rows)
    df_stats.to_csv(Path(output_dir) / "dataset_statistics_by_column.csv", index=False)

    global_rows = []
    for year, lst in summary_by_year.items():
        if not lst:
            raise ValueError(f"No data found for year {year}.")
        global_rows.append({
            "year": year,
            "total_vms": len(lst),
            "total_records": sum(d["count"] for d in lst),
            "date_range_start": min(d["start"] for d in lst),
            "date_range_end": max(d["end"] for d in lst),
        })
    pd.DataFrame(global_rows).to_csv(Path(output_dir) / "dataset_statistics_by_year.csv", index=False)
    print("✅ Summary tables saved to", output_dir)

def data_visualizer(
    dfs: Dict[str, pd.DataFrame],
    vm_names: List[str],
    output_dir: str
) -> None:
    base_dir = Path(output_dir)
    _ensure_dir(base_dir)

    for vm in vm_names:
        if vm not in dfs:
            raise ValueError(f"Requested VM '{vm}' not found in provided data dict.")
        df = dfs[vm].copy()
        df.index = pd.to_datetime(df.index)
        vm_dir = base_dir / vm
        _ensure_dir(vm_dir)

        _plot_time_series(df, out_path=vm_dir / "timeseries")
        hist_dir = vm_dir / "histograms"
        dist_dir = vm_dir / "distributions"
        acf_pacf_dir = vm_dir / "acf_pacf"
        decompose_dir = vm_dir / "decompositions"
        _ensure_dir(hist_dir)
        _ensure_dir(dist_dir)
        _ensure_dir(acf_pacf_dir)
        _ensure_dir(decompose_dir)

        _plot_histograms(df, out_dir=hist_dir)
        _plot_correlation(df, out_path=vm_dir / "correlation.svg")
        _plot_distributions(df, out_dir=dist_dir)
        _plot_acf_pacf(df, "CPU_USAGE_MHZ", Path(acf_pacf_dir))
        _plot_seasonal_decompositions(df, "CPU_USAGE_MHZ", Path(decompose_dir), periods=[12, 24, 84])

    print("✅ Visuals saved to", base_dir)

def split_visualizer(
    train_dfs: Dict[str, pd.DataFrame],
    val_dfs: Dict[str, pd.DataFrame],
    test_dfs: Dict[str, pd.DataFrame],
    online_dfs: Dict[str, pd.DataFrame],
    vm_names: List[str],
    output_dir: str = "report_output/splits"
) -> None:
    base = Path(output_dir)
    _ensure_dir(base)

    for vm in vm_names:
        if vm not in train_dfs:
            raise ValueError(f"VM '{vm}' not in splits.")
        fig, ax = plt.subplots(figsize=(9, 3))
        for df, label, color in [
            (train_dfs[vm], "train", "tab:blue"),
            (val_dfs[vm], "val", "tab:orange"),
            (test_dfs[vm], "test", "tab:green"),
            (online_dfs[vm], "online", "tab:red"),
        ]:
            ax.plot(df.index, df.iloc[:, 0], label=label, color=color, linewidth=0.7)
        ax.set_title(f"{vm} – chronological splits (first metric)")
        set_common_date_axis(ax, interval=4)
        format_date_axis(fig, rotation=30)
        ax.legend()
        _save_fig(fig, base / f"{vm}_splits")
    print("✅ Split visuals saved to", base)

def anomaly_comparison_visualizer(
    original_dfs: Dict[str, pd.DataFrame],
    reduced_dfs: Dict[str, pd.DataFrame],
    vm_names: List[str],
    output_dir: str = "report_output/anomaly_comparison"
) -> None:
    base = Path(output_dir)
    _ensure_dir(base)

    for vm in vm_names:
        if vm not in original_dfs or vm not in reduced_dfs:
            raise ValueError(f"VM '{vm}' missing in one of the provided dicts.")
        orig = original_dfs[vm]; red = reduced_dfs[vm]
        orig.index = pd.to_datetime(orig.index)
        red.index = pd.to_datetime(red.index)

        for metric in orig.columns:
            if 'is_anomaly' in metric:
                continue

            if metric not in red.columns:
                raise ValueError(f"Metric '{metric}' not found in reduced data for VM '{vm}'.")

            fig, ax = plt.subplots(figsize=(9, 3))

            ax.plot(orig.index, orig[metric], label="original", linestyle="None", marker="o", markersize=2, alpha=0.7)
            ax.plot(red.index, red[metric], label="reduced", linestyle="None", marker="o", markersize=2, alpha=0.7)

            ax.set_ylabel("CPU usage [MHz]")
            set_common_date_axis(ax, interval=4)
            format_date_axis(fig, rotation=30)
            ax.legend()

            _save_fig(fig, base / f"{vm}_{metric}_orig_vs_red")

    print("✅ Anomaly comparison plots saved to", base)
