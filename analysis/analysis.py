import numpy as np
import pandas as pd

def score_sueq(sueq_df):
    """
    Calculates the scores for the S_UEQ questionnaire.

    Parameters:
    - sueq_df (pd.DataFrame): DataFrame containing S_UEQ data (columns C-J).

    Returns:
    - dict: A dictionary with the average score for each UX scale.
    """
    # Define the mapping of scales to columns (0-based indices)
    scales = {
        'Attractiveness': [2],        # Column C
        'Perspicuity': [3, 4],        # Columns D, E
        'Efficiency': [5, 6],          # Columns F, G
        'Dependability': [7, 8],       # Columns H, I
        'Stimulation': [9],            # Column J
        # 'Novelty': [10, 11],         # Columns K, L (Out-of-Bounds for sueq_df)
    }

    scores = {}
    for scale, cols in scales.items():
        # Check if all column indices exist in sueq_df
        if max(cols) >= sueq_df.shape[1]:
            print(f"Warning: Column indices {cols} for scale '{scale}' are out of bounds.")
            scores[scale] = None
            continue
        # Calculate the average score for the scale
        scores[scale] = sueq_df.iloc[:, cols].mean(axis=1).mean()
    return scores

def score_sus(sus_df):
    """
    Calculates the SUS score from the SUS questionnaire.

    Parameters:
    - sus_df: DataFrame containing SUS data (columns K-T).

    Returns:
    - The overall SUS score (0–100 scale).
    """
    # SUS has 10 items; adjust indices if needed
    positive_items = [1, 3, 5, 7, 9]
    negative_items = [2, 4, 6, 8, 10]

    # Subtract 1 for positive items, reverse score for negatives
    sus_scores = sus_df.copy()
    sus_scores.iloc[:, positive_items] -= 1
    sus_scores.iloc[:, negative_items] = 5 - sus_scores.iloc[:, negative_items]

    # Sum and scale
    sus_total = sus_scores.sum(axis=1) * 2.5  # Scale to 0–100
    return sus_total.mean()  # Average SUS score


import pandas as pd

def score_tlx(tlx_df):
    """
    Calculates the mean score for NASA-TLX (Raw TLX) for each user.
    
    Parameters:
    - tlx_df (pd.DataFrame): DataFrame containing NASA-TLX data with the following columns:
        - First column: User ID
        - Second column: Interface Category
        - Remaining columns: TLX items (e.g., U-Z)
    
    Returns:
    - pd.DataFrame: A DataFrame with three columns:
        1. 'User ID'
        2. 'Interface Category'
        3. 'Mean TLX Score' (average across TLX items)
    """
    # Verify that tlx_df has at least three columns (ID, Group, and at least one TLX item)
    if tlx_df.shape[1] < 3:
        raise ValueError("Input DataFrame must have at least three columns: 'ID', 'Group', and TLX items.")
    
    # Identify the columns
    id_col = tlx_df.columns[0]
    group_col = tlx_df.columns[1]
    tlx_items = tlx_df.columns[2:]
    
    # Ensure TLX items are numeric, coercing errors to NaN
    numeric_tlx = tlx_df[tlx_items].apply(pd.to_numeric, errors='coerce')
    
    # Calculate the mean TLX score per user, skipping NaN values
    mean_scores = numeric_tlx.mean(axis=1, skipna=True)
    
    # Create the resulting DataFrame
    result_df = pd.DataFrame({
        'User ID': tlx_df[id_col],
        'Interface Category': tlx_df[group_col],
        'Mean TLX Score': mean_scores
    })
    
    return result_df

def visualize_questionnaire_boxplots(
    df,
    id_col='User ID',
    category_col='Interface Category',
    title='Questionnaire Item Scores by Interface Category',
    palette='Set2',
    figsize=(12, 8),
    save_path=None
):
    """
    Generates boxplots for each questionnaire item, grouped by interface category.

    Parameters:
    - df (pd.DataFrame): DataFrame containing questionnaire data.
    - id_col (str): Column name for User ID. Default is 'User ID'.
    - category_col (str): Column name for Interface Category. Default is 'Interface Category'.
    - title (str): Title of the plot. Default is 'Questionnaire Item Scores by Interface Category'.
    - palette (str or dict): Color palette for the boxplots. Default is 'Set2'.
    - figsize (tuple): Size of the matplotlib figure. Default is (12, 8).
    - save_path (str): Path to save the plot. If None, the plot is not saved. Default is None.

    Returns:
    - None. Displays the plot and optionally saves it.
    """
    # Validate that the necessary columns exist
    if id_col not in df.columns:
        raise ValueError(f"'{id_col}' column not found in the DataFrame.")
    if category_col not in df.columns:
        raise ValueError(f"'{category_col}' column not found in the DataFrame.")
    
    # Identify questionnaire items (all columns except ID and Category)
    item_cols = [col for col in df.columns if col not in [id_col, category_col]]
    
    if not item_cols:
        raise ValueError("No questionnaire items found in the DataFrame.")
    
    # Melt the DataFrame to long format for easier plotting with seaborn
    df_melted = df.melt(
        id_vars=[id_col, category_col],
        value_vars=item_cols,
        var_name='Item',
        value_name='Score'
    )
    
    # Convert 'Score' to numeric, coercing errors to NaN
    df_melted['Score'] = pd.to_numeric(df_melted['Score'], errors='coerce')
    
    # Drop rows with NaN scores
    df_melted.dropna(subset=['Score'], inplace=True)
    
    # Set the aesthetic style of the plots
    sns.set(style="whitegrid")
    
    # Initialize the matplotlib figure
    plt.figure(figsize=figsize)
    
    # Create the boxplot
    ax = sns.boxplot(
        x='Item',
        y='Score',
        hue=category_col,
        data=df_melted,
        palette=palette
    )
    
    # Set plot title and labels
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('Questionnaire Item', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    
    # Rotate x-axis labels if there are many items
    plt.xticks(rotation=45, ha='right')
    
    # Adjust legend title
    plt.legend(title=category_col, bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Tight layout for better spacing
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    # Display the plot
    plt.show()
