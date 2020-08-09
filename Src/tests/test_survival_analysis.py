import unittest

from survival_analysis import SurvivalAnalysis
import pandas as pd
from numpy import nan


# noinspection PyMethodMayBeStatic
class SurvivalAnalysisTests(unittest.TestCase):

    def setUp(self):
        """ We test on a trivial constructed dataframe.  In fact no data will be read
        outwith this class and passed in as a dataframe."""

        self.sa = SurvivalAnalysis('TCGA_OV_VST', 'AOCS_Protein', saveplots=False)

        # Make a trivial dataframe to plot
        cols_dict = {'PatientID': ['P1', 'P2', 'P3', 'P4', 'P5'],
                     'NMF_1_of_3': [0.5, 0.3, 0.6, 0.7, 0.2],
                     'NMF_2_of_3': [0.2, 0.3, 0.4, 0.7, 0.2],
                     'NMF_3_of_3': [0.4, 0.8, 0.7, 0.5, 0.2],
                     'ICA_1_of_2': [0.6, 0.8, 0.3, 0.5, 0.1],
                     'ICA_2_of_2': [0.7, 0.4, 0.9, 0.6, 0.1],
                     'donor_survival_time': [1000, 1200, 800, 1100, nan],
                     'os_event': [0, 0, 1, 1, 1],
                     'Age': [63, 54, 68, 69, 70]  # an irrelevant column
                     }
        self.df = pd.DataFrame.from_dict(cols_dict)
        self.df.set_index('PatientID', inplace=True)
        self.survival_df = self.sa.cleanup_and_threshold_components_df(self.df)

    def test_plot_unstratified_survival(self):
        self.sa.plot_unstratified_survival(self.survival_df, show=False)

    def test_run_coxs_proportional_hazards(self):
        self.sa.run_once_coxs_proportional_hazards(self.survival_df, ['NMF_1_of_3'])

    def test_cleanup_and_threshold_components_df(self):
        survival_thresholded_df = self.sa.cleanup_and_threshold_components_df(self.df)
        print(survival_thresholded_df.describe())

    def test_plot_component_stratified_survival(self):
        facto_name = 'NMF'
        nc = 3
        for comp in ['%s_%d_of_%d' % (facto_name, i, nc) for i in range(1, nc + 1)]:
            self.sa.plot_component_stratified_survival(self.survival_df, comp, show=False)

    def test_plot_hr_bars(self):
        thedict = {"PCA_1_of_3": {"TCGA(OR)": (1.2, 0.1),
                                  "AOCS(OR)": (0.9, 0.2),
                                  "AOCS(PFS)": (1.4, 0.06)},
                   "PCA_2_of_3": {"TCGA(OR)": (1.1, 0.01),
                                  "AOCS(OR)": (1.3, 0.02),
                                  "AOCS(PFS)": (1.1, 0.09)}}

        SurvivalAnalysis.plot_hr_bars(thedict, show=False)


if __name__ == '__main__':
    unittest.main()
