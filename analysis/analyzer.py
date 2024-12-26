import pandas as pd

class AnalysisManager:
    def __init__(self, sueq: pd.DataFrame, sus: pd.DataFrame, tlx: pd.DataFrame):
        """
        Initialize the DataManager with the path to the csv file.

        Args: Path to the csv file.
        """
        self.sueq = sueq
        self.sus = sus
        self.tlx = tlx

    def score_sueq(self):
        """
        Calculates the scores for the S_UEQ questionnaire.

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
            # 'Novelty': [10, 11],         # Columns K, L (Out-of-Bounds for self.sueq)
        }

        scores = {}
        for scale, cols in scales.items():
            # Check if all column indices exist in self.sueq
            if max(cols) >= self.sueq.shape[1]:
                print(f"Warning: Column indices {cols} for scale '{scale}' are out of bounds.")
                scores[scale] = None
                continue
            # Calculate the average score for the scale
            scores[scale] = self.sueq.iloc[:, cols].mean(axis=1).mean()
        return scores


    def score_tlx(self):
        """
        Calculates the mean score for NASA-TLX (Raw TLX) for each user.
        
        Parameters:
        - self.tlx (pd.DataFrame): DataFrame containing NASA-TLX data with the following columns:
            - First column: User ID
            - Second column: Interface Category
            - Remaining columns: TLX items (e.g., U-Z)
        
        Returns:
        - pd.DataFrame: A DataFrame with three columns:
            1. 'User ID'
            2. 'Interface Category'
            3. 'Mean TLX Score' (average across TLX items)
        """
        # Verify that self.tlx has at least three columns (ID, Group, and at least one TLX item)
        if self.tlx.shape[1] < 3:
            raise ValueError("Input DataFrame must have at least three columns: 'ID', 'Group', and TLX items.")
        
        # Identify the columns
        id_col = self.tlx.columns[0]
        group_col = self.tlx.columns[1]
        tlx_items = self.tlx.columns[2:]
        
        # Ensure TLX items are numeric, coercing errors to NaN
        numeric_tlx = self.tlx[tlx_items].apply(pd.to_numeric, errors='coerce')
        
        # Calculate the mean TLX score per user, skipping NaN values
        mean_scores = numeric_tlx.mean(axis=1, skipna=True)
        
        # Create the resulting DataFrame
        result_df = pd.DataFrame({
            'User ID': self.tlx[id_col],
            'Interface Category': self.tlx[group_col],
            'Mean TLX Score': mean_scores
        })
        
        return result_df


