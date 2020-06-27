import os
import unittest
from unittest import TestCase

import numpy as np

from factor_clustering import FactorClustering
from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer


class TestFactorClustering(TestCase):
    def setUp(self):
        self._clustering = None

    def clustering(self):
        if self._clustering is None:
            self._clustering = FactorClustering('Mini_Expression', 10, 'bootstrap')
            self._clustering.read_expression_matrix()
        return self._clustering

    def test_read_expression_matrix(self):
        fc = self.clustering()
        assert fc.expression_df is not None
        assert fc.expression_matrix is not None

    def test_cached_factor_repeats_filename(self):
        fc1 = FactorClustering('dummy', 10, 'bootstrap')
        pickle_fname = fc1.cached_factor_repeats_filename(NMF_Factorizer, 5)
        print(pickle_fname)
        assert 'NMF' in pickle_fname
        assert 'bootstrap' in pickle_fname

        fc2 = FactorClustering('dummy', 10, 'fixed')
        pickle_fname = fc2.cached_factor_repeats_filename(NMF_Factorizer, 5)
        print(pickle_fname)
        assert 'NMF' in pickle_fname
        assert 'fixed' in pickle_fname

    def test_compute_and_cache_one_factor_repeats(self):
        n_components = 4

        def one_test(facto_class, method, expect_randomness):
            fc = FactorClustering(self.clustering().basename, 10, method)
            fc.read_expression_matrix()
            pkl_fname = fc.compute_and_cache_one_factor_repeats(facto_class, n_components)
            assert os.path.exists(pkl_fname)
            metagene_list = fc.read_cached_factors(facto_class, n_components)
            # Ensure there is randomness in the repeat results!
            if expect_randomness:
                assert not np.array_equal(metagene_list[0], metagene_list[1])
            else:
                assert np.array_equal(metagene_list[0], metagene_list[1])
            metagene_list_2 = fc.read_cached_factors(facto_class, n_components)
            assert len(metagene_list_2) == fc.n_repeats

        one_test(NMF_Factorizer, 'bootstrap', expect_randomness=True)
        one_test(PCA_Factorizer, 'bootstrap', expect_randomness=True)
        one_test(PCA_Factorizer, 'fixed', expect_randomness=False)

    def test_single_factor_scatter(self):
        fc = self.clustering()
        n_components = 3
        facto = ICA_Factorizer
        fc.compute_and_cache_one_factor_repeats(facto, n_components, force=False)

        fc.single_factor_scatter(facto, n_components, show=False)

    def test_combined_factors_scatter(self):
        fc = self.clustering()
        fc.compute_and_cache_multiple_factor_repeats(4, 5, force=False)
        fc.combined_factors_scatter(4, show=True)

    def test_plot_multiple_combined_factors_scatter(self):
        n_components_range = 2, 4
        fc = self.clustering()
        fc.compute_and_cache_multiple_factor_repeats(*n_components_range, force=False)
        fc.plot_multiple_combined_factors_scatter(*n_components_range, show=True)

    def test_investigate_cluster_statistics(self):
        n_components = 3
        fc = self.clustering()
        fc.compute_and_cache_one_factor_repeats(NMF_Factorizer, n_components)
        result = fc.investigate_cluster_statistics(NMF_Factorizer, n_components)
        print(result)

    def test_compute_silhouette_score_and_median(self):
        n_components = 3
        facto_class = NMF_Factorizer
        fc = self.clustering()
        fc.compute_and_cache_one_factor_repeats(facto_class, n_components, force=False)
        score, median_metagenes = fc.compute_silhouette_score_and_median(
            NMF_Factorizer, n_components, doprint=False)
        print("Score = %8.6f" % score)
        assert 0 <= score <= 1.0
        assert median_metagenes.shape == (fc.n_genes, n_components)

    def test_save_multiple_median_metagenes_to_factors(self):
        fc = self.clustering()
        fc.save_multiple_median_metagenes_to_factors(NMF_Factorizer, start=2, end=5)

    def test_find_best_n_components(self):
        n_components_range = 2, 4
        facto_class = NMF_Factorizer
        fc = self.clustering()
        fc.compute_and_cache_multiple_factor_repeats(*n_components_range, force=False)
        fc.find_best_n_components(facto_class, *n_components_range, doprint=True, doshow=False)


if __name__ == '__main__':
    unittest.main()
