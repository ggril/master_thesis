import pandas as pd

def process_dataframe(df):
    """
    Processes the input pandas DataFrame by performing the following operations:
    1. Deletes the last column.
    2. Keeps the first row unchanged.
    3. Converts the first column to string type (excluding the first row).
    4. Converts the second column to categorical type (excluding the first row).
    5. Converts all remaining columns to numeric types (excluding the first row).
    
    Parameters:
    df (pd.DataFrame): The input DataFrame to process.
    
    Returns:
    pd.DataFrame: The processed DataFrame with the first row unchanged.
    """
    # Ensure the DataFrame has at least two columns
    if df.shape[1] < 2:
        raise ValueError("DataFrame must have at least two columns.")
    
    # 1. Delete the last column
    df = df.iloc[:, :-1].copy()
    
    # 2. Separate the first row
    first_row = df.iloc[[0]].copy()
    remaining_df = df.iloc[1:].copy()
    
    # 3. Change the first column to string type in remaining DataFrame
    first_col = remaining_df.columns[0]
    remaining_df[first_col] = remaining_df[first_col].astype(str)
    
    # 4. Change the second column to categorical type in remaining DataFrame
    second_col = remaining_df.columns[1]
    remaining_df[second_col] = remaining_df[second_col].astype('category')
    
    # 5. Change remaining columns to numeric types in remaining DataFrame
    remaining_cols = remaining_df.columns[2:]
    for col in remaining_cols:
        remaining_df[col] = pd.to_numeric(remaining_df[col], errors='coerce')  # Converts non-convertible values to NaN
    
    # 6. Combine the first row with the processed remaining DataFrame
    # The first row has the last column already removed
    processed_df = pd.concat([first_row, remaining_df], ignore_index=True)
    
    return processed_df



def split_data(df):
    """
    Splits the input DataFrame into three separate DataFrames representing three questionnaires after dropping the first row.

    Each resulting DataFrame contains:
    - The first two columns from the original DataFrame.
    - Specific ranges of additional columns:
        - `sueq_df`: Columns 3 to 10 (0-based indices 2 to 10).
        - `sus_df`: Columns 11 to 20 (0-based indices 10 to 20).
        - `tlx_df`: Columns 21 to 26 (0-based indices 20 to 26).

    Parameters:
    - df (pd.DataFrame): The input DataFrame to split.

    Returns:
    - tuple: A tuple containing three DataFrames (sueq_df, sus_df, tlx_df).

    Raises:
    - ValueError: If the input DataFrame does not have at least 26 columns.
    """

    # 1. Drop the first row 
    df_dropped = df.iloc[1:].copy()

    # 2. Verify that the DataFrame has at least 26 columns
    required_columns = 26
    actual_columns = df_dropped.shape[1]
    if actual_columns < required_columns:
        raise ValueError(f"Input DataFrame must have at least {required_columns} columns, but has {actual_columns} columns.")

    # 3. Select the first two columns
    first_two_cols = df_dropped.iloc[:, 0:2]

    # 4. Split into `sueq_df`: Columns 3 to 10 (indices 2 to 10)
    sueq_cols = df_dropped.iloc[:, 2:10]
    sueq_df = pd.concat([first_two_cols, sueq_cols], axis=1).reset_index(drop=True)

    # 5. Split into `sus_df`: Columns 11 to 20 (indices 10 to 20)
    sus_cols = df_dropped.iloc[:, 10:20]
    sus_df = pd.concat([first_two_cols, sus_cols], axis=1).reset_index(drop=True)

    # 6. Split into `tlx_df`: Columns 21 to 26 (indices 20 to 26)
    tlx_cols = df_dropped.iloc[:, 20:26]
    tlx_df = pd.concat([first_two_cols, tlx_cols], axis=1).reset_index(drop=True)

    return sueq_df, sus_df, tlx_df
