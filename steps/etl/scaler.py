from zenml import step
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd
from typing import Literal, Union

ScalerType = Union[StandardScaler, MinMaxScaler, RobustScaler]

def get_scaler(method: str) -> ScalerType:
    if method == "standard":
        return StandardScaler()
    elif method == "minmax":
        return MinMaxScaler()
    elif method == "robust":
        return RobustScaler()
    else:
        raise ValueError(f"Unsupported scaler method: {method}")

@step
def scaler(
    dfs: dict[str, pd.DataFrame],
    scaling_mode: Literal["per_feature", "per_feature_per_vm"],
    scaler_method: Literal["standard", "minmax", "robust"]) -> tuple[dict[str, pd.DataFrame], dict]:
    scaled_dfs = {}
    scalers = {}
    """
    #TODO: implement for train and test data split
    if scaling_mode == "global":
        all_data = pd.concat(dfs.values())
        scaler = get_scaler(scaler_method).fit(all_data)
        scalers["global"] = scaler

        for name, df in dfs.items():
            scaled_dfs[name] = pd.DataFrame(
                scaler.transform(df),
                columns=df.columns,
                index=df.index
            )

    el
    if scaling_mode == "per_vm":
        for name, df in dfs.items():
            scaler = get_scaler(scaler_method).fit(df)
            scalers[name] = scaler
            scaled_dfs[name] = pd.DataFrame(
                scaler.transform(df),
                columns=df.columns,
                index=df.index
            )

    el"""
    if scaling_mode == "per_feature":
        features = dfs[list(dfs.keys())[0]].columns
        scalers["per_feature"] = {}
        feature_scalers = scalers["per_feature"]

        for feature in features:
            all_feature_values = pd.concat([df[[feature]] for df in dfs.values()])
            scaler = get_scaler(scaler_method).fit(all_feature_values)
            feature_scalers[feature] = scaler

        for name, df in dfs.items():
            df_scaled = pd.DataFrame(index=df.index)
            for col in df.columns:
                df_scaled[col] = feature_scalers[col].transform(df[[col]]).flatten()
            scaled_dfs[name] = df_scaled

    elif scaling_mode == "per_feature_per_vm":
        for name, df in dfs.items():
            scalers[name] = {}
            scaled_df = pd.DataFrame(index=df.index)

            for col in df.columns:
                scaler = get_scaler(scaler_method).fit(df[[col]])
                scalers[name][col] = scaler
                scaled_df[col] = scaler.transform(df[[col]]).flatten()

            scaled_dfs[name] = scaled_df
    else:
        raise ValueError(f"Unsupported scaling mode: {scaling_mode}")

    return scaled_dfs, scalers
