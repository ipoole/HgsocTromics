import unittest
from heatmaps import Heatmaps
import pandas as pd


class HeatmapsTests(unittest.TestCase):

    def setUp(self):
        meta_cols = ['Outcome', 'Age']
        self.hm = Heatmaps(meta_cols)

        # Make a trivial dataframe to plot
        cols_dict = {'PatientID': ['Fred', 'Harry', 'Bert'],
                     'NMF_1_of_3': [0.5, 0.3, 0.6],
                     'NMF_2_of_3': [0.2, 0.3, 0.4],
                     'NMF_3_of_3': [0.9, 0.8, 0.7],
                     'Outcome': [1, 0, 1],
                     'Age': [63, 54, 68]}
        self.joined_df = pd.DataFrame.from_dict(cols_dict)
        self.joined_df.set_index('PatientID', inplace=True)

    def test_plot_heatmap(self):
        meta_cols = ['Outcome', 'Age']
        hm = Heatmaps(meta_cols)

        # Make a trivial dataframe to plot
        cols_dict = {'PatientID': ['Fred', 'Harry', 'Bert'],
                     'NMF_1_of_3': [0.5, 0.3, 0.6],
                     'NMF_2_of_3': [0.2, 0.3, 0.4],
                     'NMF_3_of_3': [0.9, 0.8, 0.7],
                     'Outcome': [1, 0, 1],
                     'Age': [63, 54, 68]}
        joined_df = pd.DataFrame.from_dict(cols_dict)
        joined_df.set_index('PatientID', inplace=True)

        hm.plot_heatmap(self.joined_df, show=False)


if __name__ == '__main__':
    unittest.main()
