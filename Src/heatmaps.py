""" Heatmap plotting, using Seaborn
"""

import os

import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

from join_metasamples_metadata import JoinMetasamplesMetadata


class Heatmaps:

    def __init__(self, columns_of_interest, fig_fname=None):

        self.plots_dir = '../Plots/HeatMaps/'
        self.fig_fname = fig_fname
        os.makedirs(self.plots_dir, exist_ok=True)
        self.columns_of_interest = columns_of_interest

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

        # Add some columns which will be sorted with the clustering, but to not contribute to it.
        if not self.columns_of_interest:
            target_df = None
        else:
            target_df = df[[]].copy()
            for col in self.columns_of_interest:
                target_df[col] = make_colour_col(df[col])
            target_df['-'] = [(1, 1, 1, 1)] * len(target_df)  # make a white break column

        # Plot
        fig = sns.clustermap(data, metric="correlation", standard_scale=1, row_colors=target_df,
                             cbar_pos=None)
        plt.xlabel('Metasamples')
        plt.ylabel('Patient')

        # There are too many patients to label, so turn off y labels.
        fig.ax_heatmap.set_yticks([])

        if self.fig_fname:
            figpath = self.plots_dir + 'clustered_heatmap_%s.pdf' % self.fig_fname
            print("Saving figure to", figpath)
            fig.savefig(figpath, bbox_inches='tight')

        if show:
            plt.show()


def run_one(train_basename, eval_basename, columns_of_interest, saveplots):
    # Construct a dataframe indexed by patients, with columns for
    # metasamples and metadata
    jmsmd = JoinMetasamplesMetadata(train_basename, eval_basename)
    df = jmsmd.make_joined_df()

    # Plot the heatmap, saving plots with names according to datasets
    fig_fname = 'clustered_heatmap_%s_%s.pdf' % (train_basename, eval_basename) \
        if saveplots else None
    hm = Heatmaps(columns_of_interest, fig_fname)
    hm.plot_heatmap(df)


def main():
    aocs_cols = ['WGD', 'Cellularity', 'HRDetect', 'Mutational_load', 'CNV_load',
                 'donor_survival_time']
    run_one('TCGA_OV_VST', 'AOCS_Protein', aocs_cols, saveplots=True)
    run_one('TCGA_OV_VST', 'TCGA_OV_VST', [], saveplots=True)


if __name__ == '__main__':
    main()
