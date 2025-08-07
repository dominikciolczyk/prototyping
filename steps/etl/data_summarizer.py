from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller
from river import preprocessing
from typing import Dict, Tuple
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from utils.visualization_consistency import set_plot_style


from zenml.logger import get_logger


logger = get_logger(__name__)

DATE_FORMAT = "%d-%m-%Y"

COLUMNS_TO_SUMMARIZE = [
    "CPU_USAGE_MHZ",
    "MEMORY_USAGE_KB",
    "NODE_1_DISK_IO_RATE_KBPS",
    "NODE_1_NETWORK_TR_KBPS",
]

COLUMNS_TO_SUMMARIZE = [
    "CPU_USAGE_MHZ",
    "NODE_1_DISK_IO_RATE_KBPS",
    "NODE_1_NETWORK_TR_KBPS",
]

COLUMNS_TO_YLABEL = {
    "CPU_USAGE_MHZ": "wykorzystanie CPU",
    "MEMORY_USAGE_KB": "zużycie pamięci",
    "NODE_1_DISK_IO_RATE_KBPS": "operacje dyskowe",
    "NODE_1_NETWORK_TR_KBPS": "transfer sieciowy",
}

COLUMNS_TO_YLABEL = {
    "CPU_USAGE_MHZ": "wykorzystanie CPU",
    "NODE_1_DISK_IO_RATE_KBPS": "operacje dyskowe",
    "NODE_1_NETWORK_TR_KBPS": "transfer sieciowy",
}


COLUMNS_TO_YUNITS = {
    "CPU_USAGE_MHZ": "[MHz]",
    "MEMORY_USAGE_KB": "[KB]",
    "NODE_1_DISK_IO_RATE_KBPS": "[KB/s]",
    "NODE_1_NETWORK_TR_KBPS": "[KB/s]",
}

COLUMNS_TO_YUNITS = {
    "CPU_USAGE_MHZ": "[MHz]",
    "NODE_1_DISK_IO_RATE_KBPS": "[KB/s]",
    "NODE_1_NETWORK_TR_KBPS": "[KB/s]",
}

set_plot_style()


def get_column_fullname(col: str) -> str:
    return f"{COLUMNS_TO_YLABEL[col]} {COLUMNS_TO_YUNITS[col]}"

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

def plot_metric(ax: plt.Axes, df: pd.DataFrame, col: str, label: str = None) -> None:
    sns.lineplot(
        x=df.index,
        y=df[col],
        ax=ax,
        label=label,
        linewidth=1,
        marker='o',
        markersize=3,
        linestyle='--',
    )

def _plot_time_series(df: pd.DataFrame, out_path: Path) -> None:
    for col in COLUMNS_TO_YLABEL:
        fig, ax = plt.subplots(figsize=(9, 3))

        plot_metric(ax, df, col)

        ax.set_xlabel("czas pomiaru (interwał dwugodzinny)")
        ax.set_ylabel(get_column_fullname(col))

        set_common_date_axis(ax, interval=4)
        format_date_axis(fig, rotation=0)

        _save_fig(fig, out_path.with_name(f"{out_path.stem}_{col}"))

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
    column_full = get_column_fullname(column)

    for period in periods:
        decomposition = seasonal_decompose(df[column_full].dropna(), period=period, model='additive')

        fig, axes = plt.subplots(4, 1, figsize=(10, 6), sharex=True)

        axes[0].plot(decomposition.observed.index, decomposition.observed, linewidth=1)
        axes[0].set_ylabel("dane oryginalne")

        axes[1].plot(decomposition.trend.index, decomposition.trend, linewidth=1)
        axes[1].set_ylabel("trend")

        axes[2].plot(decomposition.seasonal.index, decomposition.seasonal, linewidth=1)
        axes[2].set_ylabel("sezonowość")

        axes[3].plot(decomposition.resid.index, decomposition.resid, linestyle='None', marker='o', markersize=3)
        axes[3].set_ylabel("reszta")
        axes[3].set_xlabel("czas")

        for ax in axes:
            set_common_date_axis(ax, interval=4)

        format_date_axis(fig, rotation=0)

        fig.suptitle(column_full, fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.97])

        _save_fig(fig, output_path / f"{column_full}_decomposition_period_{period}")


def scale_dfs_per_vm(
    dfs: Dict[str, pd.DataFrame]
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, preprocessing.StandardScaler]]]:
    scaled_dfs: Dict[str, pd.DataFrame] = {}
    scalers: Dict[str, Dict[str, preprocessing.StandardScaler]] = {}

    for vm, df in dfs.items():
        vm_scalers: Dict[str, preprocessing.StandardScaler] = {}
        scaled_df = pd.DataFrame(index=df.index)

        for col in df.columns:
            scaler = preprocessing.StandardScaler()
            scaler.learn_many(df[[col]])
            scaled_col = scaler.transform_many(df[[col]]).iloc[:, 0]

            scaled_df[col] = scaled_col
            vm_scalers[col] = scaler

        scaled_dfs[vm] = scaled_df
        scalers[vm] = vm_scalers

    return scaled_dfs, scalers


def plot_correlation_heatmap(dfs: Dict[str, pd.DataFrame], out_path: Path) -> None:
    combined = pd.concat([
        df[COLUMNS_TO_SUMMARIZE] for df in dfs.values()
    ])
    combined = combined.rename(columns=COLUMNS_TO_YLABEL)

    corr = combined.corr()

    fig, ax = plt.subplots(figsize=(6, 5))
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
    _save_fig(fig, out_path / "correlation_matrix")

def run_adf_test(df: pd.DataFrame, vm_name: str) -> pd.DataFrame:
    result_rows = []
    for col, label in COLUMNS_TO_YLABEL.items():
        if col in df.columns:
            series = df[col].dropna()
            if len(series) < 10:
                raise ValueError(f"Not enough data points for ADF test on {vm_name} - {label}. Minimum 10 required.")
            else:
                stat, pval, *_ = adfuller(series, autolag='AIC', regression='ct')
                stationary = "TAK" if pval < 0.05 else "NIE"
            result_rows.append({
                "VM": vm_name,
                "Cecha": label,
                "ADF stat": stat,
                "p-value": pval,
                "Stacjonarna": stationary
            })
    return pd.DataFrame(result_rows)

def generate_stationarity_report(dfs: Dict[str, pd.DataFrame], output_path: Path) -> None:
    report = pd.concat([
        run_adf_test(df, vm_name)
        for vm_name, df in dfs.items()
    ])

    report.to_csv("report_output/stationarity_check.csv", index=False)

    report["ADF stat"] = report["ADF stat"].round(3)
    report["p-value"] = report["p-value"].round(3)

    report_pivot = report.pivot(index="VM", columns="Cecha", values="Stacjonarna").reset_index()

    desired_order = ["VM", "wykorzystanie CPU", "zużycie pamięci", "operacje dyskowe", "transfer sieciowy"]
    existing_cols = [col for col in desired_order if col in report_pivot.columns]
    report_pivot = report_pivot[existing_cols]

    def df_to_latex_table(df_subset, caption, label):
        df_copy = df_subset.copy()
        if "VM" in df_copy.columns:
            df_copy["VM"] = df_copy["VM"].apply(lambda x: x.replace("_", "\\_"))
        df_latex = df_copy.to_latex(index=False,
                                    column_format="|l|" + "c|" * (df_copy.shape[1] - 1),
                                    caption=caption,
                                    label=label,
                                    escape=False)
        lines = df_latex.splitlines()
        for i in range(6, len(lines)):
            if not lines[i].startswith("\\"):
                lines[i] = lines[i] + r" \hline"
        return "\n".join(lines)

    output_path.parent.mkdir(exist_ok=True, parents=True)
    with open(output_path, "w") as f:
        f.write(df_to_latex_table(
            report_pivot,
            caption="Ocena stacjonarności poszczególnych cech dla każdej maszyny wirtualnej (test ADF)",
            label="tab:stationarity-summary"
        ))

    return report_pivot

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

    scaled_dfs, _ = scale_dfs_per_vm(dfs)

    plot_correlation_heatmap(scaled_dfs, Path(output_dir))


def _check_stationarity(df: pd.DataFrame, out_path: Path) -> None:
    out_lines = ["Kolumna\tADF stat\tp-value\tStacjonarna\n"]
    for col in df.columns:
        series = df[col].dropna()
        try:
            result = adfuller(series)
            adf_stat, p_value = result[0], result[1]
            is_stationary = "TAK" if p_value < 0.05 else "NIE"
            out_lines.append(f"{col}\t{adf_stat:.3f}\t{p_value:.3f}\t{is_stationary}\n")
        except Exception as e:
            out_lines.append(f"{col}\tERROR\tERROR\t{str(e)}\n")

    out_path.write_text("".join(out_lines), encoding="utf-8")

def data_visualizer(
    dfs: Dict[str, pd.DataFrame],
    vm_names: List[str],
    output_dir: str
) -> None:
    base_dir = Path(output_dir)
    _ensure_dir(base_dir)

    generate_stationarity_report(dfs, Path(f"{base_dir}/stationarity_report.tex"))

    for vm in vm_names:
        if vm not in dfs:
            raise ValueError(f"Requested VM '{vm}' not found in provided data dict.")
        df = dfs[vm].copy()
        df.index = pd.to_datetime(df.index)
        vm_dir = base_dir / vm
        _ensure_dir(vm_dir)

        _check_stationarity(df, out_path=vm_dir / "stationarity.txt")

        _plot_time_series(df, out_path=vm_dir / "timeseries")
        dist_dir = vm_dir / "distributions"
        acf_pacf_dir = vm_dir / "acf_pacf"
        decompose_dir = vm_dir / "decompositions"
        _ensure_dir(dist_dir)
        _ensure_dir(acf_pacf_dir)
        _ensure_dir(decompose_dir)

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

        column = train_dfs[vm].columns[0]

        for split_df in [train_dfs[vm], val_dfs[vm], test_dfs[vm], online_dfs[vm]]:
            split_df.index = pd.to_datetime(split_df.index)

        col_label = get_column_fullname(column)

        fig, ax = plt.subplots(figsize=(9, 3))
        for df, label in [
            (train_dfs[vm], "treningowy"),
            (val_dfs[vm], "walidacyjny"),
            (test_dfs[vm], "testowy"),
            (online_dfs[vm], "online"),
        ]:
            plot_metric(ax, df, col=column, label=label)

        ax.set_xlabel("czas pomiaru (interwał dwugodzinny)")
        ax.set_ylabel(col_label)

        set_common_date_axis(ax, interval=4)
        format_date_axis(fig, rotation=0)

        #all_times = pd.concat([train_dfs[vm], val_dfs[vm], test_dfs[vm], online_dfs[vm]]).index
        #ax.set_xlim(all_times.min(), all_times.max())

        ax.legend()
        fig.tight_layout()
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

        orig = original_dfs[vm]
        red = reduced_dfs[vm]
        orig.index = pd.to_datetime(orig.index)
        red.index = pd.to_datetime(red.index)

        for metric in orig.columns:
            if 'is_anomaly' in metric:
                continue

            if metric not in red.columns:
                raise ValueError(f"Metric '{metric}' not found in reduced data for VM '{vm}'.")

            fig, ax = plt.subplots(figsize=(9, 3))

            plot_metric(ax, orig, metric, label="oryginalne")
            plot_metric(ax, red, metric, label="po redukcji")

            ax.set_xlabel("czas pomiaru (interwał dwugodzinny)")
            ax.set_ylabel(get_column_fullname(metric))
            set_common_date_axis(ax, interval=4)
            format_date_axis(fig, rotation=0)
            ax.legend()

            _save_fig(fig, base / f"{vm}_{metric}_orig_vs_red")

    print("✅ Anomaly comparison plots saved to", base)

