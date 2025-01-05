import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt

class AnalysisManager:
    def __init__(self, sueq: pd.DataFrame, sus: pd.DataFrame, tlx: pd.DataFrame):
        """
        Initialize the DataManager with the path to the csv file.

        Args: Path to the csv file.
        """
        self.sueq = sueq
        self.sus = sus
        self.tlx = tlx

    def score_ueq_short(
        self,
        id_col_index=0,
        category_col_index=1,
        pragmatic_start=2,
        pragmatic_end=6,   # Non-inclusive in iloc, so columns 2,3,4,5
        hedonic_start=6,
        hedonic_end=10     # Non-inclusive in iloc, so columns 6,7,8,9
    ):
        """
        Calculates the Short UEQ scores for each user using column indexes.
        
        Parameters:
        - id_col_index (int): Column index for User ID. Default is 0.
        - category_col_index (int): Column index for Interface Category. Default is 1.
        - pragmatic_start (int): Starting column index for Pragmatic Quality items. Default is 2.
        - pragmatic_end (int): Ending column index for Pragmatic Quality items (non-inclusive). Default is 6.
        - hedonic_start (int): Starting column index for Hedonic Quality items. Default is 6.
        - hedonic_end (int): Ending column index for Hedonic Quality items (non-inclusive). Default is 10.
        
        Returns:
        - pd.DataFrame: DataFrame with User ID, Interface Category, overall pragmatic quality score,
                        overall hedonic quality score, and overall UEQ score.
        """
        # Validate DataFrame has enough columns
        expected_columns = hedonic_end
        if self.sueq.shape[1] < expected_columns:
            raise ValueError(f"The DataFrame does not have enough columns. Expected at least {expected_columns}, got {self.sueq.shape[1]}.")

        # Extract User ID and Interface Category
        user_ids = self.sueq.iloc[:, id_col_index]
        interface_categories = self.sueq.iloc[:, category_col_index]
        
        # Extract Pragmatic and Hedonic items using iloc
        pragmatic_items_df = self.sueq.iloc[:, pragmatic_start:pragmatic_end].copy()
        hedonic_items_df = self.sueq.iloc[:, hedonic_start:hedonic_end].copy()
        
        # Ensure all questionnaire items are numeric
        pragmatic_items_df = pragmatic_items_df.apply(pd.to_numeric, errors='coerce')
        hedonic_items_df = hedonic_items_df.apply(pd.to_numeric, errors='coerce')
        
        # Handle missing values by dropping rows with any NaN in questionnaire items
        combined_df = pd.concat([user_ids, interface_categories, pragmatic_items_df, hedonic_items_df], axis=1)
        combined_df.dropna(inplace=True)
        
        # Re-extract after dropping NaNs
        user_ids = combined_df.iloc[:, id_col_index]
        interface_categories = combined_df.iloc[:, category_col_index]
        pragmatic_items_df = combined_df.iloc[:, pragmatic_start:pragmatic_end]
        hedonic_items_df = combined_df.iloc[:, hedonic_start:hedonic_end]
        
        # Calculate overall scores for each user
        overall_pragmatic = pragmatic_items_df.mean(axis=1)
        overall_hedonic = hedonic_items_df.mean(axis=1)
        overall_ueq = (overall_pragmatic + overall_hedonic) / 2
        
        # Create the resulting DataFrame
        scored_df = pd.DataFrame({
            'User ID': user_ids,
            'Interface Category': interface_categories,
            'Overall Pragmatic Quality': overall_pragmatic,
            'Overall Hedonic Quality': overall_hedonic,
            'Overall UEQ Score': overall_ueq
        })
        
        return scored_df

    def score_sus(self):
        """
        Calculates the SUS score from the SUS questionnaire.
    
        Returns:
        - pd.DataFrame: A DataFrame with 'User ID', 'Interface Category', and 'Mean SUS Score'.
        """
        # SUS has 10 items; adjust indices if needed
        positive_items = [0, 2, 4, 6, 8]  # 0-based indexing for items 1,3,5,7,9
        negative_items = [1, 3, 5, 7, 9]  # 0-based indexing for items 2,4,6,8,10
    
        # Create a copy to avoid SettingWithCopyWarning
        sus_scores = self.sus.copy()
    
        # Identify ID and Group columns
        id_col = self.sus.columns[0]
        group_col = self.sus.columns[1]
    
        # Select SUS items as a DataFrame
        # Assuming SUS items are from the 3rd column onward (0-based index 2)
        sus_items_df = sus_scores.iloc[:, 2:]
    
        # Ensure SUS items are numeric, coercing errors to NaN
        sus_items_df = sus_items_df.apply(pd.to_numeric, errors='coerce')
    
        # Adjust scores for positive items: subtract 1
        sus_items_df.iloc[:, positive_items] = sus_items_df.iloc[:, positive_items] - 1
    
        # Adjust scores for negative items: reverse score
        sus_items_df.iloc[:, negative_items] = 5 - sus_items_df.iloc[:, negative_items]
    
        # Calculate the total SUS score per user
        sus_total = sus_items_df.sum(axis=1) * 2.5  # Scale to 0–100
    
        # Create the resulting DataFrame with per-user SUS scores
        calculated_sus = pd.DataFrame({
            'User ID': sus_scores[id_col],
            'Interface Category': sus_scores[group_col],
            'Mean SUS Score': sus_total
        })
    
        return calculated_sus

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

    def combine_scores(self, ueq_scores, sus_scores, tlx_scores):
        """
        Combines the scores from the UEQ, SUS, and TLX into a single DataFrame.
        
        Parameters:
        - ueq_scores (pd.DataFrame): DataFrame with UEQ scores.
        - sus_scores (pd.DataFrame): DataFrame with SUS scores.
        - tlx_scores (pd.DataFrame): DataFrame with TLX scores.
        
        Returns:
        - pd.DataFrame: A DataFrame with combined scores.
        """
        # Merge the UEQ and SUS scores
        merged_scores = pd.merge(ueq_scores, sus_scores, on=['User ID', 'Interface Category'], how='outer')
        
        # Merge the TLX scores
        combined_scores = pd.merge(merged_scores, tlx_scores, on=['User ID', 'Interface Category'], how='outer')
        
        return combined_scores
    
    @staticmethod
    def visualize_scores(df):
        """
        Visualizes the average scores by interface category using barplots.
        
        Parameters:
            df (pd.DataFrame): The combined dataframe containing questionnaire scores.
        
        Returns:
            None: Displays a barplot of the average scores.
        """
        # Group by interface category and calculate mean for each score column
        # Rename Interface values using map
        df = df.copy()
        df['Interface Category'] = df['Interface Category'].map({"1": 'Garmin', "2": 'Strava'})
        mean_scores = df.groupby('Interface Category')[['Overall Pragmatic Quality', 
                                                        'Overall Hedonic Quality', 
                                                        'Overall UEQ Score', 
                                                        'Mean SUS Score', 
                                                        'Mean TLX Score']].mean()
        # Plot bar plots for each questionnaire score
        mean_scores.plot(kind='bar', figsize=(10, 6))
        plt.title("Average Scores by Interface")
        plt.ylabel("Average Score")
        plt.xticks(rotation=0)
        plt.legend(title="Scores")
        plt.tight_layout()
        plt.show()


    @staticmethod
    def t_stat(df):
        """
        Performs t-tests for questionnaire scores based on interface category and formats the results.
        
        Parameters:
            df (pd.DataFrame): The combined dataframe containing questionnaire scores.
        
        Returns:
            pd.DataFrame: A formatted DataFrame with t-statistics and p-values for each score column.
        """
        # Ensure 'Interface Category' exists and is numeric
        if 'Interface Category' not in df.columns:
            raise ValueError("Column 'Interface Category' is missing from the DataFrame.")
        df['Interface Category'] = pd.to_numeric(df['Interface Category'], errors='coerce')
        
        # Split the dataframe by interface category
        interface_1 = df[df['Interface Category'] == 1]
        interface_2 = df[df['Interface Category'] == 2]
        
        # Check for empty groups
        if interface_1.empty or interface_2.empty:
            raise ValueError("One or both interface categories contain no data.")
        
        # Perform t-tests for each score column
        t_results = []
        score_columns = ['Overall Pragmatic Quality', 'Overall Hedonic Quality', 
                        'Overall UEQ Score', 'Mean SUS Score', 'Mean TLX Score']
        
        for column in score_columns:
            if column not in df:
                raise ValueError(f"Column '{column}' is missing from the DataFrame.")
            
            # Handle missing data
            group1 = interface_1[column].dropna()
            group2 = interface_2[column].dropna()
            
            if group1.empty or group2.empty:
                t_results.append([column, float('nan'), float('nan')])
            else:
                t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=False, nan_policy='omit')
                t_results.append([column, t_stat, p_value])
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(t_results, columns=['Measure', 'T-Statistic', 'P-Value'])
        results_df['T-Statistic'] = results_df['T-Statistic'].round(4)
        results_df['P-Value'] = results_df['P-Value'].round(4)
        
        return results_df
