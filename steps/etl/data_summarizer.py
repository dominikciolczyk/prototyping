from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COLUMNS_TO_SUMMARIZE = [
    "CPU_USAGE_MHZ",
    "MEMORY_USAGE_KB",
    "NODE_1_DISK_IO_RATE_KBPS",
    "NODE_1_NETWORK_TR_KBPS",
]

def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)

def _plot_time_series(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3))
    df.plot(ax=ax, linewidth=1)
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.legend(fontsize=6, ncol=len(df.columns)//3 + 1)
    _save_fig(fig, out_path)

def _plot_histograms(df: pd.DataFrame, title: str, out_dir: Path) -> None:
    for col in df.columns:
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.histplot(df[col].dropna(), ax=ax, bins=30, kde=False)
        ax.set_title(f"{title} – {col}")
        _save_fig(fig, out_dir / f"hist_{col}.svg")

def _plot_correlation(df: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(df.corr(), annot=False, cmap="viridis", ax=ax)
    ax.set_title(f"Correlation – {title}")
    _save_fig(fig, out_path)

def data_summarizer(
    dfs: Dict[str, pd.DataFrame],
    output_dir: str = "report_output"
) -> None:
    """Generate LaTeX-friendly summary tables (wide format) for each VM."""
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
            "start": df.index.min().strftime("%Y-%m-%d"),
            "end": df.index.max().strftime("%Y-%m-%d"),
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
    df_stats.to_csv(Path(output_dir) / "tabela_1_wide.csv", index=False)

    global_rows = []
    for year, lst in summary_by_year.items():
        if not lst:
            continue
        global_rows.append({
            "year": year,
            "total_vms": len(lst),
            "total_records": sum(d["count"] for d in lst),
            "date_range_start": min(d["start"] for d in lst),
            "date_range_end": max(d["end"] for d in lst),
        })
    pd.DataFrame(global_rows).to_csv(Path(output_dir) / "tabela_2_latex.csv", index=False)
    print("✅ Summary tables saved to", output_dir)

def data_visualizer(
    dfs: Dict[str, pd.DataFrame],
    vm_names: List[str],
    output_dir: str = "report_output/visuals_raw"
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

        _plot_time_series(df, title=f"{vm} – Time series", out_path=vm_dir / "timeseries.svg")
        hist_dir = vm_dir / "histograms"
        _ensure_dir(hist_dir)
        _plot_histograms(df, title=vm, out_dir=hist_dir)
        _plot_correlation(df, title=vm, out_path=vm_dir / "correlation.svg")

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
        ax.legend()
        _save_fig(fig, base / f"{vm}_splits.svg")
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

        metric = orig.columns[0]
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(orig.index, orig[metric], label="original", alpha=0.6)
        ax.plot(red.index, red[metric], label="reduced", alpha=0.8)
        ax.set_title(f"{vm} – {metric}: original vs reduced")
        ax.legend()
        _save_fig(fig, base / f"{vm}_orig_vs_red.svg")

    print("✅ Anomaly comparison plots saved to", base)
