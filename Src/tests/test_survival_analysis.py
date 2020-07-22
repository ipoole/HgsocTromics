import unittest
import numpy as np
from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer
from survival_analysis import SurvivalAnalysis


# noinspection PyMethodMayBeStatic
class SurvivalAnalysisTestsTcgaTcga(unittest.TestCase):

    def setUp(self):
        self.sa = SurvivalAnalysis('TCGA_OV_VST', 'TCGA_OV_VST', saveplots=False)
        self.survival_df = self.sa.make_survival_df('NMF', 3)

    def test_compute_H_from_W_and_X(self):
        def onetest(facto_class):
            g = 50
            n = 10
            k = 3
            X = np.random.randn(g, n) + 1.0
            if facto_class == NMF_Factorizer:
                X[X < 0] = 0  # Ensure +ve

            facto = facto_class(n_components=k, tol=0.001)
            facto.fit(X)
            W = facto.get_W()

            H_computed = self.sa.compute_H_from_W_and_X(W, X, facto_class.__name__[:3])
            assert H_computed.shape == (k, n)
            print("max diff:", np.max(np.abs(H_computed - facto.get_H())))
            # We get less close results for NMF
            tol = 0.01 if facto_class == NMF_Factorizer else 1E-6
            assert np.allclose(facto.get_H(), H_computed, atol=tol)
            if facto_class == NMF_Factorizer:
                assert np.all(H_computed >= 0)

            # Check it works for a single patient
            H_computed_0 = self.sa.compute_H_from_W_and_X(W, X[:, [0]], facto_class.__name__[:3])
            assert H_computed_0.shape == (k, 1)
            assert np.allclose(H_computed_0, facto.get_H()[:, [0]], atol=tol)

        onetest(NMF_Factorizer)
        onetest(ICA_Factorizer)
        onetest(PCA_Factorizer)

    # The following will not run out of the box until factors have been
    # computed; they are bad tests!

    def test_make_survival_df(self):
        print(self.survival_df)
        print(self.survival_df.columns)

    def test_plot_unstratified_survival(self):
        self.sa.plot_unstratified_survival(self.survival_df, show=False)

    def test_run_coxs_proportional_hazards(self):
        self.sa.run_once_coxs_proportional_hazards(
            self.sa.threshold_components_df(self.survival_df), ['NMF_1_of_3'])

    def test_make_combined_survival_df(self):
        df = self.sa.make_combined_survival_df()
        cols = df.columns
        assert len(set(cols)) == len(cols)  # check no duplicates
        all_components = [c for c in cols if c not in [self.sa.time_colname, self.sa.event_colname]]
        print(all_components)
        assert len(all_components) >= 3

    def test_threshold_components_df(self):
        survival_thresholded_df = self.sa.threshold_components_df(self.survival_df)
        print(survival_thresholded_df.describe())

    def test_plot_component_stratified_survival(self):
        facto_name = 'NMF'
        nc = 3
        survival_thresholded_df = self.sa.threshold_components_df(self.survival_df)
        for comp in ['%s_%d_of_%d' % (facto_name, i, nc) for i in range(1, nc + 1)]:
            self.sa.plot_component_stratified_survival(survival_thresholded_df, comp, show=False)

    def test_plot_hr_bars(self):
        thedict = {"PCA_1_of_3": {"TCGA(OR)": (1.2, 0.1),
                                  "AOCS(OR)": (0.9, 0.2),
                                  "AOCS(PFS)": (1.4, 0.06)},
                   "PCA_2_of_3": {"TCGA(OR)": (1.1, 0.01),
                                  "AOCS(OR)": (1.3, 0.02),
                                  "AOCS(PFS)": (1.1, 0.09)}}

        SurvivalAnalysis.plot_hr_bars(thedict, show=False)


class SurvivalAnalysisTestsTcgaAocs(SurvivalAnalysisTestsTcgaTcga):
    # This version exercises cross application; metagenes from TCGA applied to AOCS
    def setUp(self):
        self.sa = SurvivalAnalysis('TCGA_OV_VST', 'AOCS_Protein')
        self.survival_df = self.sa.make_survival_df('NMF', 3)


if __name__ == '__main__':
    unittest.main()
