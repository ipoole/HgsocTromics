""" Heatmap plotting, using Seaborn
"""

import os

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.optimize import nnls
from factor_clustering import FactorClustering

import seaborn as sns


class Heatmaps:

    def __init__(self, train_basename, eval_basename, saveplots=False):
        tcga = 'TCGA_OV_VST'
        aocs = 'AOCS_Protein'

        assert train_basename in [tcga, aocs]
        assert eval_basename in [tcga, aocs]

        self.train_basename = train_basename
        self.eval_basename = eval_basename

        self.plots_dir = '../Plots/HeatMaps/'
        self.saveplots = saveplots
        os.makedirs(self.plots_dir, exist_ok=True)

        if eval_basename == tcga:
            # TCGA
            self.patient_id_colname = 'RNAID'
            self.columns_of_interest = []
        else:
            # AOCS
            self.patient_id_colname = 'Sample'
            self.columns_of_interest = ['WGD', 'Cellularity', 'HRDetect', 'Mutational_load',
                                        'CNV_load', 'donor_survival_time']

    # noinspection PyPep8Naming
    @staticmethod
    def compute_H_from_W_and_X(W, X, which_method):
        """ The matrices are in bioinformatics format, so W is (genes, factors)
        and X is (genes, patients)"""
        g_w, k = W.shape
        g_x, n = X.shape
        assert g_w == g_x
        if which_method == 'NMF':
            # Enforce +ve results.
            # nnls() does not provide the convenience of accepting a matrix b argument
            H = np.array([nnls(W, b)[0] for b in X.T]).T
        else:
            H = lstsq(W, X)[0]

        assert H.shape == (k, n)
        return H

    # noinspection PyPep8Naming
    def read_metadata_and_x_dfs(self):
        """ Create a dataframe with patients in rows, RNAID as index and interesting metadata.
        Ensure the rows are in the same order as the columns in the expression data.
        Return the metadata df and the expression df"""

        meta_fname = '../Data/%s/%s_Metadata.tsv' % (self.eval_basename, self.eval_basename)
        meta_df = pd.read_csv(meta_fname, sep='\t', index_col=self.patient_id_colname,
                              usecols=[self.patient_id_colname] + self.columns_of_interest)
        rows_from_meta = meta_df.index.values
        x_fname = '../Data/%s/%s_PrunedExpression.tsv' % (self.eval_basename, self.eval_basename)
        x_df = pd.read_csv(x_fname, sep='\t', index_col=0)
        columns_from_X = list(x_df.columns)
        assert len(columns_from_X) > 1
        assert set(rows_from_meta) == set(columns_from_X)

        # Ensure the metadata is in same order as the columns from X
        meta_df = meta_df.reindex(columns_from_X)
        return meta_df, x_df

    def read_factors_df(self, x_df, facto_prefix, nc, component_list=None):
        """ Make a dataframe of the given H factors computed from the metagene (W) factor
        and expression matrix.  Df is indexed by patient ID """

        if component_list is None:
            component_list = range(1, nc + 1)  # 1-based naming of components

        X = np.asarray(x_df)
        w_df = FactorClustering.read_median_metagenes(self.train_basename, facto_prefix, nc)
        assert list(w_df.index.values) == list(x_df.index.values)

        W = np.asarray(w_df)
        H = Heatmaps.compute_H_from_W_and_X(W, X, facto_prefix)

        # plt.hist(H.T, bins=40)
        # plt.show()
        assert H.shape == (nc, X.shape[1])

        # Copy the metagenes into the metadata

        factor_df = pd.DataFrame()
        factor_df[self.patient_id_colname] = x_df.columns
        factor_df.set_index(self.patient_id_colname, inplace=True)
        n = H.shape[1]  # number of patients
        for k in component_list:
            colname = '%s_%d_of_%d' % (facto_prefix, k, nc)
            assert len(factor_df) == n
            factor_df[colname] = H[k - 1, :]

        assert list(factor_df.index.values) == list(factor_df.index.values)

        # Clean-up: keep only the columns we need

        # df_clean = meta_df.dropna()
        # n_lost = len(meta_df) - len(df_clean)
        # # print("Number of subjects lost due to nan: ", n_lost)
        # assert n_lost / len(meta_df) < 0.1  # No more than 10% lost

        return factor_df

    def make_combined_df(self):
        """ The puts together a dataframe indexed by patient (RNAID) with metatada
        and *selected* metagenes (components).  These are the metagenes selected based
        on the cluster analysis as being stable
        """

        meta_df, x_df = self.read_metadata_and_x_dfs()
        NMF_df = self.read_factors_df(x_df, 'NMF', 3, [1, 2, 3])
        ICA_df = self.read_factors_df(x_df, 'ICA', 5, [1, 2, 3, 4, 5])
        PCA_df = self.read_factors_df(x_df, 'PCA', 3, [1, 2, 3])

        assert (list(meta_df.index.values) == list(NMF_df.index.values)
                == list(ICA_df.index.values) == list(PCA_df.index.values))

        combined_df = pd.concat([meta_df, NMF_df, ICA_df, PCA_df], axis=1)

        return combined_df

    def plot_heatmap(self, df, show=True):

        def make_colour_col(target):
            viridis = cm.get_cmap('viridis', 20)
            tmin, tmax = min(target), max(target)
            row_colors = [viridis((v - tmin) / (tmax - tmin)) for v in target]
            return row_colors

        # Pull out the metasamples which will contribute to the main heatmap
        factor_cols = [col for col in df.columns if col[:3] in ['NMF', 'ICA', 'PCA']]
        data = df[factor_cols]
        assert data.shape == (len(df), len(factor_cols))

        # Add some columns which will be sored with the clustering, but to not contribute to it.
        target_df = df[[]].copy()
        for col in self.columns_of_interest:
            target_df[col] = make_colour_col(df[col])
        target_df['-'] = [(1, 1, 1, 1)] * len(target_df)  # make a white break column

        # Plot
        fig = sns.clustermap(data, metric="correlation", standard_scale=1, row_colors=target_df)
        plt.xlabel('Metasamples')

        if self.saveplots:
            figpath = self.plots_dir + 'clustered_heatmap_%s_%s.pdf' % (
                self.train_basename[:4], self.eval_basename[:4])
            print("Saving figure to", figpath)
            fig.savefig(figpath, bbox_inches='tight')

        if show:
            plt.show()


def main():
    hm = Heatmaps('TCGA_OV_VST', 'AOCS_Protein', saveplots=True)
    df = hm.make_combined_df()
    hm.plot_heatmap(df)


if __name__ == '__main__':
    main()
