import pandas as pd
import pingouin as pg

class DataManager:
    def __init__(self, responses_data_path: str, pupil_data_path: str):
        """
        Initialize the DataManager with the path to the csv file.

        Args: Path to the csv files.
        """
        self.responses_data_path = responses_data_path
        self.pupil_data_path = pupil_data_path
        self.responses_data = None
        self.pupil_data = None

    def _load_data(self):
        """
        Load the data from the csv file.
        """
        self.responses_data = pd.read_csv(self.responses_data_path, delimiter=";")

    def _clean_data(self):
        """
        Main function to filter the data as needed from 1ka .csv file.
        Clean the data by:
        - Setting the first row as column names.
        - Removing the second row (question text).
        """
        if self.responses_data is None:
            raise ValueError("Data not loaded. Please call 'load_data' first.")

        # Drop the first row (actual questions)
        self.responses_data = self.responses_data.iloc[1:]
        self.responses_data.rename(columns={"Q1":"user_id", "Q2":"UI"}, inplace=True) 

        # Drop the last column if needed"
        if "Unnamed: 26" in self.responses_data.columns:
            self.responses_data = self.responses_data.drop(columns=["Unnamed: 26"])
            
        # Reset the index
        self.responses_data.astype('int')
        self.responses_data.reset_index(drop=True, inplace=True)

    def get_clean_data(self):
        """
        Retrieve the cleaned data.

        Returns: Cleaned pandas DataFrame.
        """
        # We load and clean data, then return dataframe
        self._load_data()
        self._clean_data()
        
        if self.responses_data is None:
            raise ValueError("Data not loaded or cleaned. Please call 'load_data' and 'clean_data' first.")

        return self.responses_data
    
    def split_data(self, idx_ids=2, idx_sueq=10, idx_sus=20, idx_tlx=26):
        """
        Splits the input DataFrame into three separate DataFrames representing three questionnaires.

        Each resulting DataFrame contains:
            - `sueq_df`: Columns 3 to 10
            - `sus_df`: Columns 11 to 20
            - `tlx_df`: Columns 21 to 26

        Returns:
        - tuple: A tuple containing three DataFrames (sueq_df, sus_df, tlx_df).

        Raises:
        - ValueError: If the input DataFrame does not have at least 26 columns.
        """
        reponses_df = self.responses_data.copy()
        
        # 2. Verify that the DataFrame has at least 26 columns
        required_columns = 26
        if len(self.responses_data.columns) != required_columns:
            raise ValueError(f"Input DataFrame must have at least {required_columns} columns, but has {len(self.responses_data.columns)} columns.")

        # 3. Select the first two columns
        first_two_cols = reponses_df.iloc[:, :idx_ids]

        # 4. Split into `sueq_df`: Columns 3 to 10 (indices 2 to 10)
        sueq_cols = reponses_df.iloc[:, idx_ids:idx_sueq]
        sueq_df = pd.concat([first_two_cols, sueq_cols], axis=1).reset_index(drop=True)

        # 5. Split into `sus_df`: Columns 11 to 20 (indices 10 to 20)
        sus_cols = reponses_df.iloc[:, idx_sueq:idx_sus]
        sus_df = pd.concat([first_two_cols, sus_cols], axis=1).reset_index(drop=True)

        # 6. Split into `tlx_df`: Columns 21 to 26 (indices 20 to 26)
        tlx_cols = reponses_df.iloc[:, idx_sus:idx_tlx]
        tlx_df = pd.concat([first_two_cols, tlx_cols], axis=1).reset_index(drop=True)

        return sueq_df, sus_df, tlx_df
    
    @staticmethod
    def calculate_cronbach_alpha(data):
        """
        Calculate Cronbach's alpha for each questionnaire and its subscales.

        Parameters:
        - data: pd.DataFrame, the dataset containing questionnaire items.

        Returns:
        - pd.DataFrame: A DataFrame with Cronbach's alpha and confidence intervals for each questionnaire.
        """
        # Define the structure of the questionnaires
        questionnaire_structure = {
            'UEQ Pragmatic Quality': ['Q4a', 'Q4b', 'Q4c', 'Q4d'],  # Replace with actual column names
            'UEQ Hedonic Quality': ['Q4e', 'Q4f', 'Q4g', 'Q4h'],    # Replace with actual column names
            'Overall UEQ Score': ['Q4a', 'Q4b', 'Q4c', 'Q4d', 'Q4e', 'Q4f', 'Q4g', 'Q4h'],  # All UEQ items
            'SUS': ['Q6a', 'Q6b', 'Q6c', 'Q6d', 'Q6e', 'Q6f', 'Q6g', 'Q6h', 'Q6i', 'Q6j'],  # SUS items
            'NASA-TLX': ['Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13']     # TLX items
        }

        results = []

        for questionnaire, items in questionnaire_structure.items():
            if not all(item in data.columns for item in items):
                raise ValueError(f"Some items for {questionnaire} are missing from the data: {items}")

            # Create a copy of the columns to avoid SettingWithCopyWarning
            subset_data = data[items].copy()

            # Ensure all columns are numeric
            subset_data = subset_data.apply(pd.to_numeric, errors='coerce')

            # Handle missing values (drop rows with NaN)
            subset_data = subset_data.dropna()

            # Calculate Cronbach's alpha
            alpha, ci = pg.cronbach_alpha(data=subset_data)

            # Append the results
            results.append({
                'Questionnaire': questionnaire,
                'Cronbach\'s Alpha': round(alpha, 4),
                'CI Lower': round(ci[0], 4),
                'CI Upper': round(ci[1], 4)
            })

        # Convert results to a DataFrame
        return pd.DataFrame(results)

    @staticmethod
    def assign_gender(dataframe, list_females):
        """
        Assign gender to a dataframe based on a list of female User IDs,
        and place the 'gender' column next to the 'user_id' column.

        Parameters:
        - dataframe: pd.DataFrame, the dataset containing the user IDs.
        - list_females: list of int, the User IDs of females.

        Returns:
        - pd.DataFrame: The updated dataframe with a new 'gender' column positioned next to 'user_id'.
        """

        # Assign gender based on the User ID
        dataframe['Gender'] = dataframe['User ID'].apply(lambda x: 'F' if x in list_females else 'M')

        # Reorder columns to place 'gender' next to 'user_id'
        columns = list(dataframe.columns)
        columns.insert(columns.index('User ID') + 1, columns.pop(columns.index('Gender')))
        dataframe = dataframe[columns]

        return dataframe.head()
