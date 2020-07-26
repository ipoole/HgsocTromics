import unittest
from unittest import TestCase

import numpy as np

from factor_clustering import FactorClustering
from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer


class TestFactorClustering(TestCase):

    def basename(self):
        return 'Mini_AOCS'

    def setUp(self):
        self._clustering = None

    def clustering(self):
        if self._clustering is None:
            self._clustering = FactorClustering(self.basename(), 10, 'bootstrap', saveplots=True)
            self._clustering.read_expression_matrix()
        return self._clustering

    def test_colour(self):
        fc = self.clustering()
        for facto in [NMF_Factorizer, ICA_Factorizer, PCA_Factorizer]:
            print(facto.__name__, fc.colour(facto))

    def test_read_expression_matrix(self):
        fc = self.clustering()
        assert fc.expression_df is not None
        assert fc.expression_matrix is not None

    def test_flip_metagenes(self):
        fc = self.clustering()
        metagenes = np.array([[1, 2, -3, 4, 5, -6, 7, 8, 9, 10],
                              [-1, 2, -3, -4, 5, 6, -7, 8, -9, -10]])
        assert metagenes.shape == (2, 10)
        flipped_metagenes = fc.flip_metagenes(metagenes)
        assert flipped_metagenes.shape == metagenes.shape
        assert np.array_equal(flipped_metagenes[0, :], metagenes[0, :])
        assert np.array_equal(flipped_metagenes[1, :], -metagenes[1, :])

    def test_cached_factor_repeats_filename(self):
        fc1 = self.clustering()
        fc1.method = 'bootstrap'
        pickle_fname = fc1.cached_factor_repeats_filename(NMF_Factorizer, 5)
        print(pickle_fname)
        assert 'NMF' in pickle_fname
        assert 'bootstrap' in pickle_fname

        fc2 = self.clustering()
        fc2.method = 'fixed'
        pickle_fname = fc2.cached_factor_repeats_filename(NMF_Factorizer, 5)
        print(pickle_fname)
        assert 'NMF' in pickle_fname
        assert 'fixed' in pickle_fname

    def test_compute_and_cache_one_factor_repeats(self):
        n_components = 4

        def one_test(facto_class, method, expect_randomness):
            fc = self.clustering()
            fc.method = method
            fc.read_expression_matrix()

            metagene_list = fc.compute_and_cache_one_factor_repeats(
                facto_class, n_components, force=True)

            # Ensure there is randomness in the repeat results!
            if expect_randomness:
                assert not np.array_equal(metagene_list[0], metagene_list[1])
            else:
                assert np.array_equal(metagene_list[0], metagene_list[1])

        one_test(NMF_Factorizer, 'bootstrap', expect_randomness=True)
        one_test(PCA_Factorizer, 'bootstrap', expect_randomness=True)
        one_test(PCA_Factorizer, 'fixed', expect_randomness=False)

    def test_compute_and_cache_multiple_factor_repeats(self):
        fc = self.clustering()
        fc.compute_and_cache_multiple_factor_repeats([4, 5], force=False)

    def test_compute_tsne_score_medians(self):
        fc = self.clustering()
        n_components = 3
        Y, score, median_metagenes = fc.compute_tsne_score_medians(ICA_Factorizer, n_components)
        assert Y.shape[0] == n_components * fc.n_repeats + n_components
        assert Y.shape[1] == 2
        assert median_metagenes.shape == (fc.n_genes, n_components)
        assert -1.0 <= score <= 1.0

    def test_plot_single_factor_scatter(self):
        fc = self.clustering()
        n_components = 3
        facto = ICA_Factorizer
        fc.plot_single_factor_scatter(facto, n_components, show=False)

    def test_compute_combined_tsne(self):
        fc = self.clustering()
        n_components = 3
        Y = fc.compute_combined_tsne(n_components)
        assert Y.shape == (3 * n_components * fc.n_repeats, 2)

    def test_plot_combined_factors_scatter(self):
        fc = self.clustering()
        fc.plot_combined_factors_scatter(4, show=False)

    def test_plot_multiple_combined_factors_scatter(self):
        nc_list = [2, 3, 4, 5]
        fc = self.clustering()
        fc.plot_multiple_combined_factors_scatter(nc_list, show=False)

    def test_plot_multiple_single_factors_scatter(self):
        nc_list = [2, 3, 4, 5]
        fc = self.clustering()
        fc.plot_multiple_single_factors_scatter(ICA_Factorizer, nc_list, show=False)

    def test_investigate_cluster_statistics(self):
        n_components = 3
        fc = self.clustering()
        result = fc.investigate_cluster_statistics(ICA_Factorizer, n_components)
        print(result)

    def test_compute_silhouette_score_and_median(self):
        n_components = 3
        facto_class = NMF_Factorizer
        fc = self.clustering()
        score, median_metagenes = fc.compute_silhouette_score_and_median(
            facto_class, n_components, doprint=False)
        print("Score = %8.6f" % score)
        assert 0 <= score <= 1.0
        assert median_metagenes.shape == (fc.n_genes, n_components)

    def test_plot_silhouette_scores(self):
        fc = self.clustering()
        fc.plot_silhouette_scores([2, 3, 4], show=False)

    def test_save_multiple_median_metagenes_to_factors(self):
        fc = self.clustering()
        fc.save_multiple_median_metagenes_to_factors(ICA_Factorizer, [3, 4])

        df = FactorClustering.read_median_metagenes(self.basename(), 'ICA', 3)
        W = np.asarray(df)
        assert W.shape == (100, 3)


class TestFactorClusteringCanon(TestFactorClustering):
    """ This repeats the tests as above but with the trivial 'Mini_Canon' dataset
    which uses different gene identifiers"""

    def basename(self):
        return 'Mini_Canon'

# Following disabled because on clean checkout we will not have the input datasets.
# Ok to enable once the TCGA and AOCS expression datasets have been copied in.
# class TestCreateIntersectionPrunedOvarianDatasets(TestCase):
#     # This is not so much a unit test, as ensuring that the pruned
#     # datasets are created on first run.  The method is lazy so no
#     # harm in repeated running.
#     @staticmethod
#     def test_create_intersection_pruned_ovarian_datasets():
#         FactorClustering.create_intersection_pruned_ovarian_datasets()


if __name__ == '__main__':
    unittest.main()
