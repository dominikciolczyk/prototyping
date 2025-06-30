from zenml import step
from typing import Union, List, Dict
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
from functools import partial
from typing import Dict, Tuple, Any
import torch
from torch.utils.data import DataLoader
from zenml import step
from zenml.logger import get_logger

logger = get_logger(__name__)

def plot_time_series(
        data: Union[dict[str, pd.DataFrame], pd.DataFrame],
        folder_name_postfix: str,
) -> list[str]:
    """
    Generate interactive time series plots (one per DataFrame or key) and save as HTML files.
    Also saves raw data (CSV) in human-readable format into a parallel 'data' folder.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    base_dir = Path("plots") / f"{timestamp}_{folder_name_postfix}"
    plots_dir = base_dir / "plots"
    data_dir = base_dir / "data"

    plots_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    output_files = []

    if isinstance(data, dict):
        for key, df in data.items():
            if df.empty:
                raise ValueError("Dataframe cannot be empty")

            # Save plot
            fig = px.line(df, x=df.index, y=df.columns, title=f"{key} Time Series")
            fig.update_layout(legend_title_text="Metrics")
            plot_path = plots_dir / f"{key}_plot.html"
            fig.write_html(str(plot_path))
            output_files.append(str(plot_path))

            # Save raw data as CSV
            data_output_path = data_dir / f"{key}.csv"
            df.to_excel(data_output_path.with_suffix(".xlsx"), index=True)

    elif isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Dataframe cannot be empty")

        # Plot
        fig = px.line(data, x=data.index, y=data.columns, title="Time Series")
        fig.update_layout(legend_title_text="Metrics")
        plot_path = plots_dir / f"plot_{timestamp}.html"
        fig.write_html(str(plot_path))
        output_files.append(str(plot_path))

        # Save raw data
        data_path = data_dir / f"data_{timestamp}.csv"
        data.to_excel(data_path.with_suffix(".xlsx"), index=True)

    return output_files

def plot_all(dfs_list: List[Dict[str, pd.DataFrame]], prefix: str):
    names = ["train", "val", "test", "test_final"]
    for name, dfs in zip(names, dfs_list):
        plot_time_series(dfs, f"{prefix}_{name}_dfs")
import matplotlib.pyplot as plt
from pathlib import Path
import imageio.v2 as imageio
import numpy as np

def _plot_step_line(
    history_df: pd.DataFrame,
    vm_id: str,
    step_idx: int,
    out_dir: str = "history_frames",
) -> Path:
    """
    Zapisuje wykres liniowy z historią aż do step_idx dla jednej VM.
    """
    out_path = Path(out_dir) / vm_id
    out_path.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=120)

    # wykrywamy kolumny 'true' (te, które nie mają _pred_model ani _pred_baseline)
    base_cols = [
        c for c in history_df.columns
        if not c.endswith("_pred_model") and not c.endswith("_pred_baseline")
    ]

    for col in base_cols:
        ax.plot(
            history_df.index,
            history_df[col],
            label=f"{col} true",
            linewidth=2
        )
        ax.plot(
            history_df.index,
            history_df[f"{col}_pred_model"],
            linestyle="--",
            label=f"{col} model",
            linewidth=1.5
        )
        ax.plot(
            history_df.index,
            history_df[f"{col}_pred_baseline"],
            linestyle=":",
            label=f"{col} baseline",
            linewidth=1.5
        )

    ax.set_title(f"{vm_id} – history up to step {step_idx}")
    ax.legend(fontsize=7, ncol=2)
    fig.autofmt_xdate()

    frame_file = out_path / f"frame_{step_idx:05d}.png"
    fig.savefig(frame_file)
    plt.close(fig)
    return frame_file

def _frames_to_gif(
    frames_dir: str,
    gif_path: str,
    fps: int = 2,
) -> None:
    """
    Łączy wszystkie PNG w katalogu frames_dir/VM_ID --> GIF.
    """
    base = Path(frames_dir)
    for vm_folder in base.iterdir():
        frames = sorted(vm_folder.glob("frame_*.png"))
        if not frames:
            continue
        imgs = [imageio.imread(str(f)) for f in frames]
        out = Path(f"{gif_path}/{vm_folder.name}_history.gif")
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out), imgs, fps=fps)
        print(f"✅ GIF saved to {out}")

