from zenml import step
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
from typing import Literal, Union, Optional

ScalerType = Union[StandardScaler, MinMaxScaler, RobustScaler, None]

def get_scaler(method: str) -> ScalerType:
    if method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "robust":
        return RobustScaler()
    elif method == "max":
        return None  # max
    else:
        raise ValueError(f"Unsupported scaler method: {method}")

@step
def scaler(
        dfs: dict[str, pd.DataFrame],
        scaler_method: Literal["standard", "minmax", "robust", "max"],
        group_scaling: bool = False,
) -> tuple[dict[str, pd.DataFrame], dict]:
    scaled_dfs = {}
    scalers = {}

    def apply_scaler(scaler_method, data: pd.DataFrame) -> pd.DataFrame:
        if scaler_method == "max":
            max_val = data.max().max()
            if max_val == 0:
                max_val = 1e-8
            return data / max_val
        else:
            scaler = get_scaler(scaler_method).fit(data)
            return pd.DataFrame(
                scaler.transform(data),
                columns=data.columns,
                index=data.index
            )

    def is_group_col(col: str) -> bool:
        return "DISK" in col.upper() or "NETWORK" in col.upper()

    for vm_name, df in dfs.items():
        df_scaled = pd.DataFrame(index=df.index)
        scalers[vm_name] = {}

        if group_scaling:
            group_cols = [col for col in df.columns if is_group_col(col)]
            indiv_cols = [col for col in df.columns if col not in group_cols]

            if group_cols:
                df_scaled[group_cols] = apply_scaler(scaler_method, df[group_cols])
                scalers[vm_name]["group"] = group_cols

            for col in indiv_cols:
                col_scaled = apply_scaler(scaler_method, df[[col]])
                df_scaled[col] = col_scaled[col]
                scalers[vm_name][col] = "individual"
        else:
            for col in df.columns:
                col_scaled = apply_scaler(scaler_method, df[[col]])
                df_scaled[col] = col_scaled[col]
                scalers[vm_name][col] = "individual"

        scaled_dfs[vm_name] = df_scaled

    return scaled_dfs, scalers
