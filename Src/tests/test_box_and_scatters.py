import unittest

import numpy as np
import pandas as pd

from box_and_scatters import BoxAndScatters


# noinspection PyMethodMayBeStatic
class BoxAndScattersTests(unittest.TestCase):

    def test_plot_boxplots(self):
        bas = BoxAndScatters(['WGD'], ['WGD'], dataset_tag='gash')

        # Make a trivial dataframe to plot
        cols_dict = {'PatientID': ['P1', 'P2', 'P3', 'P4', 'P5'],
                     'NMF_1_of_3': [0.5, 0.3, 0.6, 0.7, 0.2],
                     'NMF_2_of_3': [0.2, 0.3, 0.4, 0.7, 0.2],
                     'NMF_3_of_3': [0.4, 0.8, 0.7, 0.5, 0.2],
                     'ICA_1_of_2': [0.6, 0.8, 0.3, 0.5, 0.1],
                     'ICA_2_of_2': [0.7, 0.4, 0.9, 0.6, 0.1],
                     'WGD': [0, 0, 1, 1, 1],
                     'Age': [63, 54, 68, 69, 70]  # an irrelevant column
                     }
        df = pd.DataFrame.from_dict(cols_dict)
        df.set_index('PatientID', inplace=True)
        bas.plot_boxplots(df, 'NMF', show=False)

    def test_plot_scatters(self):
        aocs_cols = ['WGD', 'Cellularity', 'HRDetect', 'Mutational_load', 'CNV_load', 'SV_load']
        bas = BoxAndScatters(aocs_cols[:1], aocs_cols, dataset_tag='gash')
        # Make a trivial dataframe to plot
        n = 20
        cols_dict = {'PatientID': ['P%d' % (i + 1) for i in range(n)],
                     'NMF_1_of_3': np.random.randn(n),
                     'NMF_2_of_3': np.random.randn(n),
                     'NMF_3_of_3': np.random.randn(n),
                     'ICA_1_of_2': np.random.randn(n),
                     'ICA_2_of_2': np.random.randn(n),
                     'WGD': np.random.choice([0, 1], n),
                     'Cellularity': np.random.randn(n),
                     'HRDetect': np.random.randn(n),
                     'Mutational_load': np.random.randn(n),
                     'CNV_load': np.random.randn(n),
                     'SV_load': np.random.randn(n),
                     'Age': np.random.randn(n),  # an irrelevant column
                     }
        df = pd.DataFrame.from_dict(cols_dict)
        df.set_index('PatientID', inplace=True)
        df.at['P1', 'Mutational_load'] = 127424  # Test the special outlier case
        bas.plot_scatters(df, show=False)


if __name__ == '__main__':
    unittest.main()
