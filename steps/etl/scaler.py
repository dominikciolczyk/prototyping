from zenml import step
import pandas as pd
from sklearn.preprocessing import StandardScaler

@step
def scaler(dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:

    all_data = pd.concat(dfs.values())
    scaler = StandardScaler().fit(all_data)

    result =  {
        name: pd.DataFrame(
            scaler.transform(df),
            columns=df.columns,
            index=df.index
        )
        for name, df in dfs.items()
    }
    print (result)
    return result
