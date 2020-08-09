""" Boxplots and scatter plots to relate metasamples metadata
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from join_metasamples_metadata import JoinMetasamplesMetadata


# noinspection PyMethodMayBeStatic
class BoxAndScatters:

    def __init__(self, dataset_tag=None):
        self.plots_dir = '../Plots/BoxAndScatters/'
        self.fig_name = dataset_tag
        os.makedirs(self.plots_dir, exist_ok=True)

    @staticmethod
    def component_columns(df):
        """ Return a list of the factor columns (e.g. NMF_1_of_3) present in the df"""
        return [col for col in df.columns if col[:4] in ['NMF_', 'ICA_', 'PCA_']]

    def plot_boxplots(self, df, facto_prefix, colour, show=True):
        selected_columns = [col for col in df.columns if facto_prefix in col]

        x = df[selected_columns].values  # returns a numpy array
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)
        df[selected_columns] = x_scaled

        melted_df = pd.melt(df, id_vars=['WGD'], value_vars=selected_columns)

        sns.boxplot(x='variable', y='value', hue='WGD', color=colour, data=melted_df)

        if self.fig_name:
            figpath = self.plots_dir + 'wgd_boxplot_%s_%s.pdf' % (facto_prefix, self.fig_name)
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')
        if show:
            plt.show()


def run_one(train_basename, eval_basename, saveplots):

    # Construct a dataframe indexed by patients, with columns for metasamples and metadata
    df = JoinMetasamplesMetadata(train_basename, eval_basename).make_joined_df()

    # Plot the heatmaps, saving plots with names according factorizer and datasets
    fig_name = '%s_%s' % (train_basename[:4], eval_basename[:4]) if saveplots else None
    bas = BoxAndScatters(dataset_tag=fig_name)
    bas.plot_boxplots(df, 'NMF', u'#1f77b4')   # Colours follow our blue, orange, green convention
    bas.plot_boxplots(df, 'ICA', u'#ff7f0e')
    bas.plot_boxplots(df, 'PCA', u'#2ca02c')


def main():
    run_one('TCGA_OV_VST', 'AOCS_Protein', saveplots=True)


if __name__ == '__main__':
    main()
