# coding: utf-8

# # Compare factors delivered by the three methods
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.utils import resample

from factorizer_wrappers import ICA_Factorizer, NMF_Factorizer, PCA_Factorizer


# ## Factorizer classes
# These have now been hived off into ``factorizer_wrappers.py``.  Import and test them.


# noinspection PyStringFormat,PyMethodMayBeStatic
class FactorClustering:
    # ## Read the expression matrix
    # This is repeated code, should be factored out...
    def __init__(self, basename, n_repeats=50, method='bootstrap', saveplots=False):
        assert method in ['bootstrap', 'fixed']
        self.basename = basename  # Eg "Mini_Expression"
        self.gene_column_name = 'Gene_ID' if 'Canon' in self.basename else 'GeneENSG'
        self.shortname = self.basename[:4]  # e.g. "Mini"
        self.cache_dir = '../Cache/%s/FactorClustering/' % self.basename
        self.plots_dir = '../Plots/%s/FactorClustering/' % self.basename
        self.method = method
        self.saveplots = saveplots
        self.n_repeats = n_repeats
        self.expression_df = None
        self.expression_matrix = None
        self.expression_filename = None
        self.n_genes = None
        self.n_patients = None
        self.gene_naming = None
        self.colours = {'NMF': u'#1f77b4', 'ICA': u'#ff7f0e', 'PCA': u'#2ca02c'}
        # these are from the standard matplotlib colour cycle

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)

    def colour(self, facto):
        return self.colours[facto.__name__[:3]]

    @staticmethod
    def read_pair_of_expressions_and_intersect(file_1, file_2):
        """Read expression data from both files returning a pair of dataframes having only
        genes which are present in both files"""
        df_1 = pd.read_csv(file_1, sep='\t', index_col=0)

        df_2 = pd.read_csv(file_2, sep='\t', index_col=0)

        genes_1 = set(df_1.index)
        genes_2 = set(df_2.index)
        intersection_genes = genes_1.intersection(genes_2)

        intersection_df = pd.DataFrame()
        intersection_df['GeneENSG'] = list(intersection_genes)
        intersection_df.set_index('GeneENSG', inplace=True)

        df_1_pruned = pd.merge(df_1, intersection_df, how='inner', on='GeneENSG')
        df_2_pruned = pd.merge(df_2, intersection_df, how='inner', on='GeneENSG')
        df_1_pruned.sort_index(inplace=True)
        df_2_pruned.sort_index(inplace=True)

        return df_1_pruned, df_2_pruned

    @staticmethod
    def create_intersection_pruned_ovarian_datasets():

        input_filename_1 = '../Data/AOCS_Protein/AOCS_Protein_NOT_Expression.tsv'
        input_filename_2 = '../Data/TCGA_OV_VST/TCGA_OV_VST_NOT_Expression.tsv'

        output_filename_1 = '../Data/AOCS_Protein/AOCS_Protein_PrunedExpression.tsv'
        output_filename_2 = '../Data/TCGA_OV_VST/TCGA_OV_VST_PrunedExpression.tsv'

        if not (os.path.exists(output_filename_1) and os.path.exists(output_filename_2)):
            print("Creating pruned (synchonised) versions of AOCS and TCGA datasets...")
            df_1, df_2 = FactorClustering.read_pair_of_expressions_and_intersect(
                input_filename_1, input_filename_2)

            df_1.to_csv(output_filename_1, index=True, index_label='GeneENSG', sep='\t')
            df_2.to_csv(output_filename_2, index=True, index_label='GeneENSG', sep='\t')

            print("Done.")
        else:
            print("Intersection pruned AOCS and TCGA datasets already exist.")

    def read_expression_matrix(self):
        # Read in expression spreadsheet which has been processed (see end of notebook)
        # to inlcude only protein coding genes

        self.expression_filename = '../Data/%s/%s_PrunedExpression.tsv' % (
            self.basename, self.basename)
        self.expression_df = pd.read_csv(self.expression_filename, sep='\t', index_col=0)

        self.expression_matrix = np.asarray(self.expression_df)
        if 'Canon' in self.basename:  # HACK: Canon TempO-Seq data
            clip_val = np.percentile(self.expression_matrix, 99.9)
            self.expression_matrix[self.expression_matrix > clip_val] = clip_val
        self.n_genes, self.n_patients = self.expression_matrix.shape
        print(self.n_genes, "genes")
        print(self.n_patients, "patients")

    def cached_factor_repeats_filename(self, facto_class, n_components):

        # ## Multiple cached runs of NMF and ICA
        # Run NMF and ICA for a range of components, with repeats and save into .pkl fles
        # for later use.

        pickle_fname = self.cache_dir + "%s_%d_%d_%s.pkl" % \
                       (facto_class.__name__, n_components, self.n_repeats, self.method)
        return pickle_fname

    def compute_and_cache_one_factor_repeats(self, facto_class, n_components, force=False):
        """
        For the given factorizer and n_components, perform n_repeats factorizations with different
        random seeds and resample-with-replacement patients (bootstrapping).  Save results to
        a file named for the combination of parameters.  If 'force' is False, then computation
        is skipped if the cache file already exists.
        """
        assert self.expression_matrix is not None
        assert self.method in ['bootstrap', 'fixed']
        pickle_fname = self.cached_factor_repeats_filename(facto_class, n_components)
        p = Path(pickle_fname)
        os.makedirs(p.parent, exist_ok=True)
        if force or not os.path.exists(pickle_fname):
            print('Computing %s ... ' % pickle_fname)
            metagene_list = []
            V = self.expression_matrix
            n = V.shape[1]  # number of patients

            for i in range(self.n_repeats):
                facto = facto_class(n_components=n_components,
                                    random_state=np.random.randint(10000))
                if self.method == 'bootstrap':
                    # Make a bootstrap sample. The resample() method works on rows, hence the need
                    # for transpose in and out.
                    resampled_V = resample(V.T, n_samples=n).T
                    assert resampled_V.shape == V.shape
                    facto.fit(resampled_V)
                else:
                    # fixed sampling
                    facto.fit(V)
                metagene_list += [facto.get_W()]
                print('\r%d/%d' % (i + 1, self.n_repeats), end='')
            print()
            with open(pickle_fname, 'wb') as f:
                pickle.dump(metagene_list, f)
        with open(pickle_fname, 'rb') as f:
            metagene_list = pickle.load(f)
        return metagene_list

    def compute_and_cache_multiple_factor_repeats(self, nc_list, force=False):
        # This will take several hours, if enabled!
        if True:
            one = self.compute_and_cache_one_factor_repeats
            for nc in nc_list:
                # Note that NMF and ICA require very different tolerances to work correctly,
                # and these are defined by the FactorizerWrappers.
                # Here we compute and discard the result - results will have been cached.
                _ = one(NMF_Factorizer, nc, force=force)
                _ = one(ICA_Factorizer, nc, force=force)
                _ = one(PCA_Factorizer, nc, force=force)

            print("Multiple repeat factorizations all done.")

    def compute_tsne_score_medians(self, facto_class, n_components, pca_reduced_dims=20):
        # t-SNE projection.  Also get median metagenes and score.

        facto_name = facto_class.__name__[:3]
        tsne_cache_filename = self.cache_dir + 'tsne_score_medians_%s_%d_%d_%s.pkl' % (
            facto_name, n_components, self.n_repeats, self.method)

        if not os.path.exists(tsne_cache_filename):
            metagene_list = self.compute_and_cache_one_factor_repeats(facto_class, n_components)
            score, median_metagenes = self.compute_silhouette_score_and_median(
                facto_class, n_components, pca_reduced_dims=pca_reduced_dims)

            stacked_metagenes = np.hstack(metagene_list + [median_metagenes]).T

            flipped_metagenes = self.flip_metagenes(stacked_metagenes)

            # Reduce to a managable number of dimensions before passing to t-SNE
            pca = PCA(n_components=min(pca_reduced_dims, len(flipped_metagenes)))
            tsne = TSNE(n_components=2, init='pca', n_jobs=7)
            Y = tsne.fit_transform(pca.fit_transform(flipped_metagenes))
            assert Y.shape[0] == n_components * self.n_repeats + n_components
            with open(tsne_cache_filename, 'wb') as f:
                pickle.dump((Y, score, median_metagenes), f)

        with open(tsne_cache_filename, 'rb') as f:
            Y, score, median_metagenes = pickle.load(f)

        return Y, score, median_metagenes

    def plot_single_factor_scatter(self, facto_class, n_components, show=True):
        # Plot the t-SNE projections
        # First plot the scattered points
        (Y, score, median_metagenes) = self.compute_tsne_score_medians(facto_class, n_components)
        plt.scatter(Y[:n_components * self.n_repeats, 0], Y[:n_components * self.n_repeats, 1],
                    c=self.colour(facto_class), s=3, label='one sample')

        # Then plot the medians
        plt.scatter(Y[n_components * self.n_repeats:, 0], Y[n_components * self.n_repeats:, 1],
                    c='k', s=100, marker='+', label='cluster median')

        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.legend()
        facto_name = facto_class.__name__[:3]
        plt.title("%s; %s; k=%d; silhouette s.=%6.4f" %
                  (self.shortname, facto_name, n_components, score))
        # figname = 'single_factor_scatter_%s_%s_%d_%s' % (self.shortname, facto_name, n_components,
        #                                                  self.method)
        # if self.saveplots:
        #     figpath = self.plots_dir + figname + '.pdf'
        #     print("Saving figure to", figpath)
        #     plt.savefig(figpath, bbox_inches='tight')
        if show:
            plt.show()

    def flip_metagenes(self, stacked_metagenes):
        """ For ICA there is a problem that a metagenes can be oriented 180 from each other
        but are effectively the same.  This version simply ensures the most extreme value
        is +ve"""
        return np.array([g if np.max(g) > -np.min(g) else -g for g in stacked_metagenes[:]])

    def flip_metagenes_2(self, stacked_metagenes):
        """ For ICA there is a problem that a metagenes can be oriented 180 from each other
        but are effectively the same.  This version looks at the most extreme value"""
        return np.array([g if np.max(g) > -np.min(g) else -g for g in stacked_metagenes[:]])

    def compute_combined_tsne(self, n_components, pca_reduced_dims=20):
        # ## t-SNE plots of NMF, ICA and PCA components
        # Interesting to see the components generated by the three methods ploted in the same space.
        # However, we must beware of over-interpeting t-SNE plots...

        # Read back the pickle files containing multiple runs. One file for each n_components
        # for each of NMF and ICA
        combined_tsne_cache_filename = self.cache_dir + 'combined_tsne_%d_%d_%s.pkl' % \
                                       (n_components, self.n_repeats, self.method)
        if not os.path.exists(combined_tsne_cache_filename):
            nmf_mg_list = self.compute_and_cache_one_factor_repeats(NMF_Factorizer, n_components)
            ica_mg_list = self.compute_and_cache_one_factor_repeats(ICA_Factorizer, n_components)
            pca_mg_list = self.compute_and_cache_one_factor_repeats(PCA_Factorizer, n_components)

            stacked_metagenes = np.hstack(nmf_mg_list + ica_mg_list + pca_mg_list).T

            # For ICA at least, we see double the expected number of components, due to the
            # arbitrary direction of the vector.  So flip them into the same overall direction
            flipped_metagenes = self.flip_metagenes(stacked_metagenes)

            # Reduce to a managable number of dimensions before passing to t-SNE
            pca = PCA(n_components=min(pca_reduced_dims, len(flipped_metagenes)))
            tsne = TSNE(n_components=2, init='pca', n_jobs=7)
            Y = tsne.fit_transform(pca.fit_transform(flipped_metagenes))

            # Plot the t-SNE projections in different colours for each factorizer
            assert Y.shape == (3 * n_components * self.n_repeats, 2)

            with open(combined_tsne_cache_filename, 'wb') as f:
                pickle.dump(Y, f)
        with open(combined_tsne_cache_filename, 'rb') as f:
            Y = pickle.load(f)

        return Y

    def plot_combined_factors_scatter(self, n_components, show=True):
        Y = self.compute_combined_tsne(n_components)
        Ys = np.reshape(Y, (3, n_components * self.n_repeats, 2))

        plt.scatter(Ys[0, :, 0], Ys[0, :, 1], c=self.colour(NMF_Factorizer), s=5, label='NMF')
        plt.scatter(Ys[1, :, 0], Ys[1, :, 1], c=self.colour(ICA_Factorizer), s=5, label='ICA')
        plt.scatter(Ys[2, :, 0], Ys[2, :, 1], c=self.colour(PCA_Factorizer), s=5, label='PCA')

        plt.xlabel("t-SNE dimension 1")
        plt.ylabel("t-SNE dimension 2")
        ax = plt.gca()
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

        plt.legend()
        plt.title("Components: %d" % n_components)
        figname = 'combined_factors_scatter_%s_%d_%s' % (self.shortname, n_components, self.method)
        if self.saveplots:
            figpath = self.plots_dir + figname + '.pdf'
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')
        if show:
            plt.show()

    def plot_multiple_combined_factors_scatter(self, nc_list, show=True):
        n = len(nc_list)
        cols = 4
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(5*cols, 5*rows))
        for i, nc in enumerate(nc_list):
            print('.', end='')
            plt.subplot(rows, cols, i + 1)
            self.plot_combined_factors_scatter(nc, show=False)

        plt.suptitle("%s; t-SNE clusterings for %d bootstraps of NMF, ICA and PCA" %
                     (self.shortname, self.n_repeats), size=14, y=1.0)
        nc_list_str = '_'.join([str(c) for c in nc_list])
        figname = 'multiple_combined_factors_scatter_%s_%s_%s' % (self.shortname, nc_list_str,
                                                                  self.method)
        if self.saveplots:
            figpath = self.plots_dir + figname + '.pdf'
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')
        if show:
            plt.show()

    def plot_multiple_single_factors_scatter(self, facto_class, nc_list, show=True):
        n = len(nc_list)
        cols = 4
        rows = int(np.ceil(n / cols))
        plt.figure(figsize=(5 * cols, 4.5 * rows))
        for i, nc in enumerate(nc_list):
            print('.', end='')
            plt.subplot(rows, cols, i + 1)
            self.plot_single_factor_scatter(facto_class, nc, show=False)
        plt.suptitle("%s, %s; t-SNE clustering for %d repeats, with %s sampling" %
                     (self.shortname, facto_class.__name__[:3], self.n_repeats, self.method),
                     size=16)
        facto_name = facto_class.__name__[:3]
        nc_list_str = '_'.join([str(c) for c in nc_list])
        figname = 'multiple_single_factors_scatter_%s_%s_%s_%s' % \
                  (self.shortname, facto_name, nc_list_str, self.method)
        if self.saveplots:
            figpath = self.plots_dir + figname + '.pdf'
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')
        if show:
            plt.show()

    def investigate_cluster_statistics(self, facto_class, n_components,
                                       pca_reduced_dims=20, doprint=False):
        # The given facto is not actually executed, just used to select the appropriate cached
        # .pkl files which were computed above.
        metagene_list = self.compute_and_cache_one_factor_repeats(facto_class, n_components)
        stacked_metagenes = np.hstack(metagene_list).T
        flipped_metagenes = self.flip_metagenes(stacked_metagenes)

        pca = PCA(n_components=min(pca_reduced_dims, len(metagene_list)))
        kmeans = KMeans(n_clusters=n_components, random_state=0).fit(
            pca.fit_transform(flipped_metagenes))
        cluster_table = np.reshape(kmeans.labels_, (self.n_repeats, n_components))
        clusters_are_aligned = np.all(
            [cluster_table[r, :] == cluster_table[0, :] for r in range(self.n_repeats)])
        if doprint:
            for r in range(self.n_repeats):
                print(r, cluster_table[r, :])
            print()
        return clusters_are_aligned

    def investigate_multiple_cluster_statistics(self, nc_list):
        # Lets see if clusters are assined consistently for NMF and ICA across a
        # range of n_components
        print("%6s %10s %10s %10s" %
              ('', NMF_Factorizer.__name__, ICA_Factorizer.__name__, PCA_Factorizer.__name__))
        for nc in nc_list:
            nmf_consistent = self.investigate_cluster_statistics(NMF_Factorizer, nc)
            ica_consistent = self.investigate_cluster_statistics(ICA_Factorizer, nc)
            pca_consistent = self.investigate_cluster_statistics(PCA_Factorizer, nc)
            print("%6d%10s %10s %10s" % (nc, nmf_consistent, ica_consistent, pca_consistent))

    def compute_silhouette_score_and_median(self, facto_class, n_components,
                                            pca_reduced_dims=20, doprint=False):

        # Get repeated metagenes for this n_components into a matrix of shape
        # (n_components*n_repeats, genes)
        metagene_list = self.compute_and_cache_one_factor_repeats(facto_class, n_components)
        assert metagene_list[0].shape[0] == self.n_genes
        stacked_metagenes = np.hstack(metagene_list).T
        flipped_metagenes = np.array(self.flip_metagenes(stacked_metagenes))
        assert flipped_metagenes.shape[0] == n_components * self.n_repeats

        # Run k-means clustering, but reduce to a sensible number of dimensions first
        pca = PCA(n_components=min(pca_reduced_dims, len(metagene_list)))
        kmeans = KMeans(n_clusters=n_components, random_state=0).fit(
            pca.fit_transform(flipped_metagenes))
        cluster_labels = kmeans.fit_predict(flipped_metagenes)
        cluster_label_table = np.reshape(cluster_labels, (self.n_repeats, n_components))
        silhouette_avg = silhouette_score(flipped_metagenes, cluster_labels)

        if doprint:
            for r in range(self.n_repeats):
                print(r, cluster_label_table[r, :])

            print("For n_clusters =", n_components,
                  "The average silhouette_score is :", silhouette_avg)

        # Now let's collect the samples for each cluster and find the median centroid

        median_metagenes = []
        for nc in range(n_components):
            ixs = np.where(cluster_labels == nc)[0]
            metagenes_for_this_comp = flipped_metagenes[ixs, :]
            median_metagene = np.median(metagenes_for_this_comp, axis=0)
            median_metagenes += [median_metagene]

        # Put the metagenes in form of a matrix in standard (genes, components) orientation
        median_metagenes_matrix = np.vstack(median_metagenes).T
        assert median_metagenes_matrix.shape == (self.n_genes, n_components)

        return silhouette_avg, median_metagenes_matrix

    def save_multiple_median_metagenes_to_factors(self, facto_class, nc_list):
        os.makedirs('../Factors/%s' % self.basename, exist_ok=True)
        for nc in nc_list:
            facto_prefix = facto_class.__name__[:3]
            fname = '../Factors/%s/%s_median_factor_%d.tsv' % (self.basename, facto_prefix, nc)
            _, _, median_metagenes = self.compute_tsne_score_medians(facto_class, nc)
            assert median_metagenes.shape == (self.n_genes, nc)
            # We want to write a .tsv file with ids in the first column, then
            # columns for the nc components
            columns = ['IC%d' % c for c in range(nc)]
            factor_df = pd.DataFrame(data=median_metagenes, columns=columns)
            factor_df[self.gene_column_name] = self.expression_df.index
            factor_df.set_index(self.gene_column_name, inplace=True)
            assert len(factor_df.columns) == nc
            factor_df.to_csv(fname, sep='\t')

            print('\r%d' % nc, end='')

    @staticmethod
    def read_median_metagenes(basename, facto_prefix, nc):
        fname = '../Factors/%s/%s_median_factor_%d.tsv' % (basename, facto_prefix, nc)
        df = pd.read_csv(fname, sep='\t', index_col=0)
        return df

    def plot_silhouette_scores(self, nc_list, show=True):
        plt.figure(figsize=(6, 4))
        NMF_sc, ICA_sc, PCA_sc = {}, {}, {}
        compute = self.compute_tsne_score_medians
        for nc in nc_list:
            # compute function returns (tsne, score, median), so we want [1]
            NMF_sc[nc] = compute(NMF_Factorizer, nc)[1]
            ICA_sc[nc] = compute(ICA_Factorizer, nc)[1]
            PCA_sc[nc] = compute(PCA_Factorizer, nc)[1]

        plt.plot(list(NMF_sc.keys()), list(NMF_sc.values()), '-o', c=self.colour(NMF_Factorizer),
                 label='NMF')
        plt.plot(list(ICA_sc.keys()), list(ICA_sc.values()), '-o', c=self.colour(ICA_Factorizer),
                 label='ICA')
        plt.plot(list(PCA_sc.keys()), list(PCA_sc.values()), '-o', c=self.colour(PCA_Factorizer),
                 label='PCA')
        plt.xlabel('Factorization rank (k)')
        plt.ylabel('Silhouette score')
        plt.xticks(np.arange(min(nc_list), max(nc_list), step=1))
        plt.legend()
        plt.title("%s; Silhouette scores (%s sampling)" % (self.shortname, self.method))

        nc_list_str = '_'.join([str(c) for c in nc_list])
        figname = 'silhouette_plots_%s_%s_%s' % (self.shortname, nc_list_str, self.method)
        if self.saveplots:
            figpath = self.plots_dir + figname + '.pdf'
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')
        if show:
            plt.show()


# noinspection PyUnusedLocal,PyUnreachableCode
def one_run(basename, method, saveplots=True):
    print('\n\nStarting Factor Clustering for %s, method=%s\n' % (basename, method))

    fc = FactorClustering(basename, 50, method, saveplots=saveplots)
    fc.read_expression_matrix()

    # nc_list = [2, 3, 5, 10]  # These are the ranks we'll show in the paper.
    nc_list = [3, 5]  # These are the ranks we'll show in the paper.

    if True:
        # Beware - this will take hours (for the full size dataset)!
        fc.compute_and_cache_multiple_factor_repeats(nc_list, force=False)

    if True:
        fc.plot_multiple_combined_factors_scatter(nc_list, show=False)

    if False:
        fc.investigate_multiple_cluster_statistics(nc_list)

    if True:
        fc.plot_multiple_single_factors_scatter(NMF_Factorizer, nc_list, show=False)
        fc.plot_multiple_single_factors_scatter(ICA_Factorizer, nc_list, show=False)
        fc.plot_multiple_single_factors_scatter(PCA_Factorizer, nc_list, show=False)

    if False:
        fc.plot_silhouette_scores(range(2, 11), show=False)

    if True:
        if fc.method == 'bootstrap':
            for facto_class in [NMF_Factorizer, ICA_Factorizer, PCA_Factorizer]:
                fc.save_multiple_median_metagenes_to_factors(facto_class, nc_list)


def main():

    # On first run we must generate the '..._PrunedExpression.tsv' datasets for each of TCGA
    # and AOCS.  This results in datasets with identical gene sets.  The function is lazy
    # so no harm in calling it subsequently.
    FactorClustering.create_intersection_pruned_ovarian_datasets()

    possible_datasets = {1: 'Mini_AOCS',
                         2: 'AOCS_Protein',
                         3: 'TCGA_OV_VST',
                         4: 'Canon_N200'}

    one_run(possible_datasets[1], 'bootstrap', saveplots=True)
    one_run(possible_datasets[1], 'fixed', saveplots=True)

    # one_run(possible_datasets[2], 'bootstrap', saveplots=True)
    # one_run(possible_datasets[2], 'fixed', saveplots=True)
    # #
    # one_run(possible_datasets[3], 'bootstrap', saveplots=True)
    # one_run(possible_datasets[3], 'fixed', saveplots=True)
    #
    # one_run(possible_datasets[4], 'bootstrap', saveplots=True)
    # one_run(possible_datasets[4], 'fixed', saveplots=True)

    print('\n\n All Done.')


if __name__ == '__main__':
    main()


