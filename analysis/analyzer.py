import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import pingouin as pg
import seaborn as sns

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
        # SUS has 10 items
        positive_items = [0, 2, 4, 6, 8]  # 0-based indexing for items 1,3,5,7,9
        negative_items = [1, 3, 5, 7, 9]  # 0-based indexing for items 2,4,6,8,10
    
        # Create a copy to avoid SettingWithCopyWarning
        sus_scores = self.sus.copy()
    
        # Identify ID and Group columns
        id_col = self.sus.columns[0]
        group_col = self.sus.columns[1]
    
        # Select SUS items as a DataFrame
        # Assuming SUS items are from the 3rd column onward (first is ID, second is Group)
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
    def calculate_cohen_d(df, group_col, score_columns):
        """
        Calculate Cohen's d for multiple questionnaire scores grouped by a specific column.

        Parameters:
        - df: pd.DataFrame, the dataset containing the scores and grouping column.
        - group_col: str, the column by which to group data (e.g., 'Interface Category', 'UI order', 'UI rank').
        - score_columns: list of str, columns for which to compute Cohen's d.

        Returns:
        - pd.DataFrame: A DataFrame with Cohen's d for each score and group.
        """
        results = []

        # Iterate through each score column
        for score in score_columns:
            if score not in df.columns:
                raise ValueError(f"Column '{score}' not found in the DataFrame.")

            # Check unique groups in the group_col
            unique_groups = df[group_col].unique()
            if len(unique_groups) != 2:
                raise ValueError(f"Column '{group_col}' must have exactly two groups for Cohen's d calculation.")

            # Split the data into two groups
            group1 = df[df[group_col] == unique_groups[0]][score].dropna()
            group2 = df[df[group_col] == unique_groups[1]][score].dropna()

            # Compute Cohen's d
            cohen_d = pg.compute_effsize(group1, group2, eftype='cohen')

            # Append the result
            results.append({
                'Measure': score,
                'Grouping Variable': group_col,
                'Group 1': unique_groups[0],
                'Group 2': unique_groups[1],
                'Cohen\'s d': round(cohen_d, 4),
            })

        # Convert results to a DataFrame
        return pd.DataFrame(results)
    
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
        Performs paired t-tests for questionnaire scores for dependent measures,
        aligning data based on User ID.

        Parameters:
            df (pd.DataFrame): The combined dataframe containing questionnaire scores.
                Must contain 'User ID', 'Interface Category' (1 = Interface 1, 2 = Interface 2),
                and the questionnaire score columns.

        Returns:
            pd.DataFrame: A DataFrame with t-statistics and p-values for each score column.
        """
        # Ensure required columns are present
        required_columns = ['User ID', 'Interface Category']
        score_columns = ['Overall Pragmatic Quality', 'Overall Hedonic Quality', 
                        'Overall UEQ Score', 'Mean SUS Score', 'Mean TLX Score']
        
        missing_columns = [col for col in required_columns + score_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"The following required columns are missing: {missing_columns}")

        # Ensure 'Interface Category' is numeric
        df['Interface Category'] = pd.to_numeric(df['Interface Category'], errors='coerce')

        # Pivot the data to align based on User ID
        df_pivoted = df.pivot(index='User ID', columns='Interface Category', values=score_columns)

        # Initialize results list
        t_results = []

        # Iterate through the score columns
        for column in score_columns:
            # Extract paired data for the current column
            group1 = df_pivoted[(column, 1)]  # Interface Category 1
            group2 = df_pivoted[(column, 2)]  # Interface Category 2

            # Drop NaN values for valid pairing
            paired_data = pd.concat([group1, group2], axis=1, keys=['Group1', 'Group2']).dropna()

            if paired_data.empty:
                t_stat, p_value = float('nan'), float('nan')
            else:
                # Perform paired t-test
                t_stat, p_value = stats.ttest_rel(paired_data['Group1'], paired_data['Group2'])

            # Append the results
            t_results.append({
                'Measure': column,
                'T-Statistic': round(t_stat, 4) if pd.notna(t_stat) else 'NaN',
                'P-Value': round(p_value, 4) if pd.notna(p_value) else 'NaN'
            })

        # Convert results to a DataFrame
        results_df = pd.DataFrame(t_results)
        return results_df

    def calculate_phi_coefficient(self, col1, col2):
        """
        Calculate the Phi coefficient for two binary categorical variables.

        Parameters:
        - col1: pd.Series, First binary variable
        - col2: pd.Series, Second binary variable

        Returns:
        - float: Phi coefficient
        """
        contingency_table = pd.crosstab(col1, col2)
        chi2, _, _, _ = stats.chi2_contingency(contingency_table)
        phi = np.sqrt(chi2 / len(col1))
        return phi

    def calculate_point_biserial(self, binary_col, continuous_col):
        """
        MIGHT USE SPEARMANN?
        Calculate Point-Biserial correlation between a binary variable and a continuous variable.

        Parameters:
        - binary_col: pd.Series, Binary variable
        - continuous_col: pd.Series, Continuous variable

        Returns:
        - float, float: Correlation coefficient, p-value
        """
        return stats.pointbiserialr(binary_col, continuous_col)
    
    def gender_cor(self, data, score_columns):
        """
        Calculate Point-Biserial correlations between Gender and each score column.

        Parameters:
        - data: pd.DataFrame, dataset containing the scores and Gender column.
        - score_columns: list of str, columns containing the scores.

        Returns:
        - pd.DataFrame: DataFrame showing correlations and p-values between Gender and scores.
        """

        # Map gender to binary values (F -> 1, M -> 0)
        binary_gender = data['Gender'].map({'F': 1, 'M': 0})

        # Calculate correlations for each score column
        results = []
        for score in score_columns:
            if data[score].dtype.kind not in "bifc":  # Check for numeric data
                raise ValueError(f"Score column '{score}' must be numeric.")
            corr, p_value = self.calculate_point_biserial(binary_gender, data[score])
            results.append({
                'Score Column': score.replace('_', ' ').capitalize(),
                'Correlation (r)': round(corr, 4),
                'P-Value': round(p_value, 4)
            })

        # Convert results to DataFrame
        return pd.DataFrame(results)
    

    def correlations_and_ttests(self, combined_df):
        """
        Perform correlation and paired t-test analyses for the combined dataset.

        Parameters:
        - combined_df: pd.DataFrame, Combined dataset with all questionnaire scores.

        Returns:
        - correlations: List of tuples (variable1, variable2, correlation, p-value)
        - ttests: List of tuples (measure, grouping_variable, t-statistic, p-value)
        """
        # Normalize column names to avoid issues with spaces or casing
        combined_df.columns = combined_df.columns.str.strip()

        # Column names
        rank_col = 'UI rank'
        order_col = 'UI order'
        questionnaire_columns = [
            'Overall Pragmatic Quality',
            'Overall Hedonic Quality',
            'Overall UEQ Score',
            'Mean SUS Score',
            'Mean TLX Score'
        ]

        # Lists to store results
        correlations = []
        ttests = []

        # Phi coefficient for UI rank ↔ UI order
        phi = self.calculate_phi_coefficient(combined_df[rank_col], combined_df[order_col])
        correlations.append((rank_col, order_col, phi, 'Phi coefficient (no p-value)'))

        # Point-Biserial correlations for each questionnaire score with UI rank and UI order
        for column in questionnaire_columns:
            if column in combined_df.columns:
                # Point-Biserial with UI rank
                pb_corr_rank, pb_p_rank = self.calculate_point_biserial(combined_df[rank_col], combined_df[column])
                correlations.append((rank_col, column, pb_corr_rank, pb_p_rank))

                # Point-Biserial with UI order
                pb_corr_order, pb_p_order = self.calculate_point_biserial(combined_df[order_col], combined_df[column])
                correlations.append((order_col, column, pb_corr_order, pb_p_order))

        # Perform paired t-tests for each questionnaire score based on UI rank and UI order
        for grouping_col in [rank_col, order_col]:
            for column in questionnaire_columns:
                if column not in combined_df.columns:
                    continue

                # Ensure data contains exactly one entry for each User ID and grouping variable
                valid_users = combined_df.groupby('User ID')[grouping_col].nunique()
                valid_users = valid_users[valid_users == 2].index  # Only include users with both groupings
                paired_df = combined_df[combined_df['User ID'].isin(valid_users)]

                # Separate data for grouping variable (e.g., UI rank or UI order)
                group1 = paired_df[paired_df[grouping_col] == 1][['User ID', column]].set_index('User ID')
                group2 = paired_df[paired_df[grouping_col] == 2][['User ID', column]].set_index('User ID')

                # Merge groups by User ID to ensure proper pairing
                paired_data = pd.concat([group1, group2], axis=1, keys=['Group1', 'Group2']).dropna()

                # Perform paired t-test if data is available
                if paired_data.empty:
                    t_stat, p_value = float('nan'), float('nan')
                else:
                    t_stat, p_value = stats.ttest_rel(paired_data['Group1'], paired_data['Group2'])
                    # Ensure scalar values
                    t_stat = float(t_stat) if isinstance(t_stat, np.ndarray) else t_stat
                    p_value = float(p_value) if isinstance(p_value, np.ndarray) else p_value

                # Append results
                ttests.append((column, grouping_col, t_stat, p_value))

        return correlations, ttests

    
    def present_findings(self, correlations, ttests):
        """
        Present findings in a formatted table with significance indicators.

        Parameters:
        - correlations: list of tuples, each containing (variable1, variable2, correlation, p-value)
        - ttests: list of tuples, each containing (measure, grouping_variable, t-statistic, p-value)

        Returns:
        - pd.DataFrame: Formatted DataFrames for correlations and t-tests.
        """
        # Process correlation results
        corr_results = []
        for var1, var2, corr, p_value in correlations:
            significance = ''
            if isinstance(p_value, (int, float)):  # Handle Phi coefficient (no p-value)
                if p_value < 0.05:
                    significance = '*'
                if p_value < 0.01:
                    significance = '**'
            corr_results.append({
                'Relationship': f"{var1} ↔ {var2}",
                'Correlation (r)': f"{corr:.2f}{significance}",
                'P-Value': f"{p_value:.4f}" if isinstance(p_value, (int, float)) else p_value
            })

        # Process t-test results
        ttest_results = []
        for measure, group, t_stat, p_value in ttests:
            significance = ''
            if isinstance(p_value, (int, float)):
                if p_value < 0.05:
                    significance = '*'
                if p_value < 0.01:
                    significance = '**'
            ttest_results.append({
                'Measure': measure,
                'Grouping Variable': group,
                'T-Statistic (t)': f"{t_stat:.2f}{significance}" if isinstance(t_stat, (int, float)) else "NaN",
                'P-Value': f"{p_value:.4f}" if isinstance(p_value, (int, float)) else "NaN"
            })

        # Convert to DataFrames for tabular display
        corr_df = pd.DataFrame(corr_results)
        ttest_df = pd.DataFrame(ttest_results)

        return corr_df, ttest_df
    
    @staticmethod
    def scatter_matrix(data, score_columns, independent_variable):
        """
        Create a scatterplot matrix using all combinations of scoring columns,
        with the independent variable represented as color.

        Parameters:
        - data: pd.DataFrame, the dataset containing the scores and independent variable.
        - score_columns: list of str, the scoring columns to be used as dependent variables.
        - independent_variable: str, the independent variable used for coloring the points.

        Returns:
        - None: Displays the scatterplot matrix.
        """
        sns.set_theme(style="whitegrid", color_codes=True)

        # Normalize column names
        data.columns = data.columns.str.strip().str.lower().str.replace(" ", "_")
        score_columns = [col.strip().lower().replace(" ", "_") for col in score_columns]
        independent_variable = independent_variable.strip().lower().replace(" ", "_")

        # Ensure necessary columns exist
        missing_columns = [col for col in score_columns + [independent_variable] if col not in data.columns]
        if missing_columns:
            raise KeyError(f"The following columns are missing from the dataset: {missing_columns}")

        # Create the PairGrid
        g = sns.PairGrid(data=data, vars=score_columns, height=3)
        
        # Map scatter plots with color based on the independent variable
        g.map(
            sns.scatterplot,
            hue=data[independent_variable],  # Use independent variable as color
            palette="viridis",
            alpha=0.7
        )

        # Add a legend
        g.add_legend(title=independent_variable.replace("_", " ").capitalize())
        
        # Add title
        g.fig.suptitle(
            f"Scatterplot Matrix of Scores Colored by {independent_variable.replace('_', ' ').capitalize()}",
            fontsize=16
        )
        g.fig.tight_layout()
        g.fig.subplots_adjust(top=0.93)  # Adjust title position

        plt.show()


    @staticmethod
    def visualize_scores_by(df, independent_variable):
        """
        Visualizes the average scores grouped by the desired independent variable using barplots.

        Parameters:
            df (pd.DataFrame): The combined dataframe containing questionnaire scores.
            independent_variable (str): The column by which the data should be grouped.

        Returns:
            None: Displays a barplot of the average scores.
        """
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
        independent_variable = independent_variable.strip().lower().replace(" ", "_")

        # List of score columns
        score_columns = [
            'overall_pragmatic_quality',
            'overall_hedonic_quality',
            'overall_ueq_score',
            'mean_sus_score',
            'mean_tlx_score'
        ]

        # Ensure the independent variable and score columns exist
        if independent_variable not in df.columns:
            raise KeyError(f"The independent variable '{independent_variable}' is not in the dataset.")
        missing_columns = [col for col in score_columns if col not in df.columns]
        if missing_columns:
            raise KeyError(f"The following score columns are missing from the dataset: {missing_columns}")

        # Group by the independent variable and calculate mean scores
        mean_scores = df.groupby(independent_variable)[score_columns].mean()

        # Plot the barplot
        mean_scores.plot(kind='bar', figsize=(10, 6))
        plt.title(f"Average Scores by {independent_variable.replace('_', ' ').capitalize()}")
        plt.ylabel("Average Score")
        plt.xticks(rotation=0)
        plt.legend(title="Scores", loc="upper right")
        plt.tight_layout()
        plt.show()


