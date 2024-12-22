import pandas as pd

class DataManager:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        """Loads the data from an Excel file."""
        data = pd.read_excel(self.file_path)
        return data

    def get_interface_data(self, data, interface):
        """Filters the data for a specific interface."""
        return data[data['Interface'] == interface]

    def get_combined_data(self):
        """Loads and returns all data."""
        return self.load_data()
    

print("ALO")    