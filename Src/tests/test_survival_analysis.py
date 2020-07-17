import unittest
import numpy as np
from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer
import survival_analysis


# noinspection PyMethodMayBeStatic
class MyTestCase(unittest.TestCase):

    def test_compute_H_from_W_and_X(self):
        def onetest(facto_class):
            g = 50
            n = 10
            k = 3
            X = np.random.randn(g, n) + 2.0
            X[X < 0] = 0  # Ensure +ve

            facto = facto_class(n_components=k, tol=0.001)
            facto.fit(X)
            W = facto.get_W()

            H_computed = survival_analysis.compute_H_from_W_and_X(W, X, facto_class.__name__[:3])
            assert H_computed.shape == (k, n)
            print("max diff:", np.max(np.abs(H_computed - facto.get_H())))
            # We get less close results for NMF
            tol = 0.01 if facto_class == NMF_Factorizer else 1E-6
            assert np.allclose(facto.get_H(), H_computed, atol=tol)
            if facto_class == NMF_Factorizer:
                assert np.all(H_computed >= 0)

            # Check it works for a single patient
            H_computed_0 = survival_analysis.compute_H_from_W_and_X(W, X[:, [0]],
                                                                    facto_class.__name__[:3])
            assert H_computed_0.shape == (k, 1)
            assert np.allclose(H_computed_0, facto.get_H()[:, [0]], atol=tol)

        onetest(NMF_Factorizer)
        onetest(ICA_Factorizer)
        onetest(PCA_Factorizer)

    # The following will not run out of the box until factors have been
    # computed; they are bad tests!

    # def test_make_survival_df(self):
    #     basename = 'TCGA_OV_VST'
    #     df_meta = survival_analysis.make_survival_df(basename, 'ICA', 3)
    #     print(df_meta)
    #     print(df_meta.columns)
    #
    # def test_plot_unstratified_survival(self):
    #     basename = 'TCGA_OV_VST'
    #     df_meta = survival_analysis.make_H_dataframe(basename, 'ICA', 3)
    #     survival_analysis.plot_unstratified_survival(df_meta)
    #
    # def test_run_coxs_proportional_hazards(self):
    #     basename = 'TCGA_OV_VST'
    #     df_meta = survival_analysis.make_H_dataframe(basename, 'PCA', 7)
    #     survival_analysis.run_coxs_proportional_hazards(df_meta)


if __name__ == '__main__':
    unittest.main()
