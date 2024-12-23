import numpy as np

class Analysis:
    def __init__(self):
        """
        Initializes the Analysis class.
        """
        pass

    def score_sueq(self, sueq_df):
        """
        Calculates the mean score for the S_UEQ questionnaire.

        Parameters:
        - sueq_df: DataFrame containing S_UEQ data (columns C-J).

        Returns:
        - The mean score across all S_UEQ scales.
        """
        return sueq_df.iloc[:, 1:].mean(axis=1).mean()  # Exclude the interface column, then average per user and overall


    # def score_sueq(self, sueq_df):
    #     """
    #     Calculates the scores for the S_UEQ questionnaire.

    #     Parameters:
    #     - sueq_df: DataFrame containing S_UEQ data (columns C-J).

    #     Returns:
    #     - A dictionary with the average score for each UX scale.
    #     """
    #     # Define the mapping of scales to columns
    #     scales = {
    #         'Attractiveness': [2],        # Adjust indices as per actual column positions
    #         'Perspicuity': [3, 4],
    #         'Efficiency': [5, 6],
    #         'Dependability': [7, 8],
    #         'Stimulation': [9, 10],
    #         'Novelty': [11, 12],
    #     }

    #     scores = {}
    #     for scale, cols in scales.items():
    #         # Average per user for the given scale, then overall mean
    #         scores[scale] = sueq_df.iloc[:, cols].mean(axis=1).mean()
    #     return scores



    def score_sus(self, sus_df):
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

    def score_tlx(self, tlx_df):
        """
        Calculates the average score for NASA-TLX (Raw TLX).

        Parameters:
        - tlx_df: DataFrame containing NASA-TLX data (columns U-Z).

        Returns:
        - The average workload score across all dimensions.
        """
        # Average each row (user), then overall mean
        return tlx_df.mean(axis=1).mean()
