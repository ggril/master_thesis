import pandas as pd

class DataManager:

    """ Class DataManager reads data from excel file and splits them into 3 data frames"""
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads the data from an Excel file and renames column A to 'Interface'."""
        data = pd.read_excel(self.file_path)
        data.rename(columns={data.columns[0]: 'Interface'}, inplace=True)
        return data


    def split_data(self, data):
        """
        Splits the data into three dataframes representing the questionnaires:
        - S_UEQ: Columns B (index 1) + C-J (index 2-9)
        - SUS: Columns B (index 1) + K-T (index 10-19)
        - NASA-TLX: Columns B (index 1) + U-Z (index 20-25)

        Parameters:
        - data: The loaded data DataFrame.

        Returns:
        - A dictionary with three DataFrames: {'S_UEQ': df1, 'SUS': df2, 'NASA-TLX': df3}
        """
        s_ueq = data.iloc[:, [1] + list(range(2, 10))]  # Column B + C-J
        sus = data.iloc[:, [1] + list(range(10, 20))]   # Column B + K-T
        nasa_tlx = data.iloc[:, [1] + list(range(20, 26))]  # Column B + U-Z

        return {
            'S_UEQ': s_ueq,
            'SUS': sus,
            'NASA-TLX': nasa_tlx
        }
  