import matplotlib.pyplot as plt
from scipy.stats import ttest_ind

class Analysis:
    def __init__(self, data, interface_col='Interface'):
        """
        Initializes the Analysis class.

        Parameters:
        - data: DataFrame containing the self-report data.
        - interface_col: The column name indicating which interface the user used.
        """
        self.data = data
        self.interface_col = interface_col

    def perform_ttest(self, metric, group1='Interface_1', group2='Interface_2'):
        """
        Performs a t-test for a given metric between two interfaces.

        Parameters:
        - metric: The column name of the metric to analyze.
        - group1: Name of the first group in the interface column.
        - group2: Name of the second group in the interface column.

        Returns:
        - A dictionary with the metric, t-statistic, and p-value.
        """
        group1_data = self.data[self.data[self.interface_col] == group1][metric]
        group2_data = self.data[self.data[self.interface_col] == group2][metric]

        stat, p_value = ttest_ind(group1_data, group2_data, nan_policy='omit')
        return {'metric': metric, 'stat': stat, 'p_value': p_value}

    def plot_metric(self, metric):
        """
        Generates a boxplot for a given metric across interfaces.

        Parameters:
        - metric: The column name of the metric to visualize.
        """
        self.data.boxplot(column=metric, by=self.interface_col, grid=False)
        plt.title(f'{metric} by {self.interface_col}')
        plt.suptitle('')
        plt.xlabel(self.interface_col)
        plt.ylabel(metric)
        plt.show()

    def summarize_results(self, ttest_results):
        """
        Summarizes t-test results into a printable format.

        Parameters:
        - ttest_results: List of t-test result dictionaries.
        """
        for result in ttest_results:
            print(f"Metric: {result['metric']}")
            print(f"t-statistic: {result['stat']:.2f}, p-value: {result['p_value']:.4f}\n")

    def analyze_metrics(self, metrics, group1='Interface_1', group2='Interface_2'):
        """
        Analyzes a list of metrics by performing t-tests and plotting results.

        Parameters:
        - metrics: List of metric column names to analyze.
        - group1: Name of the first group in the interface column.
        - group2: Name of the second group in the interface column.

        Returns:
        - List of t-test results for all metrics.
        """
        results = []
        for metric in metrics:
            # Perform t-test
            result = self.perform_ttest(metric, group1, group2)
            results.append(result)

            # Generate plot
            self.plot_metric(metric)

        return results