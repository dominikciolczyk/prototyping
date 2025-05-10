import pandas as pd

def concat_dataframes_horizontally(dataframes: list[pd.DataFrame]) -> pd.DataFrame:
    # Create a set to track all column names across dataframes
    all_columns = set()

    # Create a list to store the processed dataframes
    processed_dfs = []

    for i, df in enumerate(dataframes):
        new_cols = []
        for col in df.columns:
            if col in all_columns:
                new_cols.append(f'df{i}_{col}')
            else:
                new_cols.append(col)
            all_columns.add(new_cols[-1])
        df.columns = new_cols

        if not df.index.is_unique:
            df = df.groupby(df.index).mean()
        processed_dfs.append(df)

    # Concatenate the dataframes horizontally
    result = pd.concat(processed_dfs, axis=1)

    return result