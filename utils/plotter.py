from zenml import step
from typing import Union, List, Dict
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path

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
                continue

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
        if not data.empty:
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
    names = ["train", "val", "test, ""test_teacher", "test_student", "online"]
    for name, dfs in zip(names, dfs_list):
        plot_time_series(dfs, f"{prefix}_{name}_dfs")