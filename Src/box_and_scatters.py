""" Boxplots and scatter plots to relate metasamples metadata
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

from join_metasamples_metadata import JoinMetasamplesMetadata


# noinspection PyMethodMayBeStatic
class BoxAndScatters:

    def __init__(self, box_cols, scatter_cols, dataset_tag=None):
        assert len(box_cols) == 1   # Current limitation
        self.box_cols = box_cols
        self.scatter_cols = scatter_cols
        self.plots_dir = '../Plots/BoxAndScatters/'
        self.dataset_tag = dataset_tag
        self.colours = {'NMF': u'#1f77b4', 'ICA': u'#ff7f0e', 'PCA': u'#2ca02c'}
        os.makedirs(self.plots_dir, exist_ok=True)

    @staticmethod
    def component_columns(df):
        """ Return a list of the factor columns (e.g. NMF_1_of_3) present in the df"""
        return [col for col in df.columns if col[:4] in ['NMF_', 'ICA_', 'PCA_']]

    def normalise_component_columns(self, df, selected_columns):
        x = df[selected_columns].values  # returns a numpy array
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        df_result = df.copy()
        df_result[selected_columns] = x_scaled
        return df_result

    def plot_boxplots(self, df, facto_prefix, show=True):
        assert len(self.box_cols) == 1
        selected_columns = [col for col in df.columns if facto_prefix in col]
        df = self.normalise_component_columns(df, selected_columns)
        melted_df = pd.melt(df, id_vars=self.box_cols, value_vars=selected_columns)

        sns.boxplot(x='variable', y='value', hue=self.box_cols[0], color=self.colours[facto_prefix],
                    data=melted_df)

        if self.dataset_tag:
            figpath = self.plots_dir + 'wgd_boxplot_%s_%s.pdf' % (facto_prefix, self.dataset_tag)
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')
        if show:
            plt.show()

    def plot_scatters(self, df, show=True):
        factors = [col for col in df.columns if col[:3] in ['NMF', 'ICA', 'PCA']]
        df = self.normalise_component_columns(df, factors)

        rows, cols = len(factors), len(self.scatter_cols)
        plt.figure(figsize=(2.5*cols, 2*rows))
        i = 0
        for factor in factors:
            for feat in self.scatter_cols:
                i += 1
                plt.subplot(rows, cols, i)

                if feat == 'Mutational_load':
                    # Remove an outlier which messes up the scaling
                    outlier = 127424
                    x = df[feat][df[feat] < outlier]
                    y = df[factor][df[feat] < outlier]
                elif feat in ['WGD', 'Which']:
                    # WGD is 0 or 1, so jitter slightly
                    jitter = np.random.uniform(-0.1, 0.1, len(df))
                    x = df[feat]+jitter
                    y = df[factor]
                else:
                    x, y = df[feat], df[factor]
                plt.scatter(x, y, c=self.colours[factor[:3]])

                # Labels only on the left and bottom plots
                if i > (rows - 1) * cols:
                    plt.xlabel(feat, size=16)
                if i % cols == 1:
                    plt.ylabel(factor, size=16)

                # No scales - there's no space and it would not be informative
                plt.xticks([])
                plt.yticks([])

                # Calculate and show correlation coefficients and p-values
                # Must get x and y again to avoid jitter and outlier changes
                x = df[feat].values
                y = df[factor].values
                if feat in ['WGD', 'Which']:
                    # Binary value, so use Point-Biserialr correlation
                    r, p_val = stats.pointbiserialr(x, y)
                else:
                    r, p_val = stats.pearsonr(x, y)
                star = '***' if p_val <= 0.01 else ''
                annotation = 'r=%4.2f, p=%0.3f %s' % (r, p_val, star)
                plt.title(annotation)

        if self.dataset_tag:
            figpath = self.plots_dir + 'genomic_feature_scatters_%s.pdf' % self.dataset_tag
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')

        if show:
            plt.show()


def run_one(train_basename, eval_basename, box_cols, scatter_cols, saveplots):

    # Construct a dataframe indexed by patients, with columns for metasamples and metadata
    df = JoinMetasamplesMetadata(train_basename, eval_basename).make_joined_df()

    # Plot the heatmaps, saving plots with names according factorizer and datasets
    fig_name = '%s_%s' % (train_basename[:4], eval_basename[:4]) if saveplots else None
    bas = BoxAndScatters(box_cols, scatter_cols, dataset_tag=fig_name)
    bas.plot_boxplots(df, 'NMF')
    bas.plot_boxplots(df, 'ICA')
    bas.plot_boxplots(df, 'PCA')

    bas.plot_scatters(df)


def main():
    aocs_cols = ['WGD', 'Cellularity', 'HRDetect', 'Mutational_load', 'CNV_load', 'SV_load']
    run_one('TCGA_OV_VST', 'AOCS_Protein', aocs_cols[:1], aocs_cols, saveplots=True)

    run_one('BOTH_AOCS_TCGA', 'BOTH_AOCS_TCGA', ['Which'], ['Which'], saveplots=True)
    run_one('BOTH_AOCS_TCGA', 'AOCS_Protein', aocs_cols[:1], aocs_cols, saveplots=True)


if __name__ == '__main__':
    main()
