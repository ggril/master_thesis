import pandas as pd

class DataManager:
    def __init__(self, responses_data_path: str, pupil_data_path: str):
        """
        Initialize the DataManager with the path to the csv file.

        Args: Path to the csv file.
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

        # Drop the last column if it's named "Unnamed: 26"
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

