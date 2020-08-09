import unittest
from box_and_scatters import BoxAndScatters
import pandas as pd


# noinspection PyMethodMayBeStatic
class BoxAndScattersTests(unittest.TestCase):

    def test_plot_boxxplots(self):
        bas = BoxAndScatters(dataset_tag='gash')

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
        bas.plot_boxplots(df, 'NMF', colour=u'#1f77b4', show=False)


if __name__ == '__main__':
    unittest.main()
