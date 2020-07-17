""" Survival Analysis
   * Read training factors file (metagenes, W)
   * Read the training expression matrix
   * Compute H matrix by least squares
   * Read training survival data from metadata
   * Perform Cox's Hazard with H factors as predictors
"""

import numpy as np
from factor_clustering import FactorClustering
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.linalg import lstsq
from scipy.optimize import nnls


# noinspection PyPep8Naming
def compute_H_from_W_and_X(W, X, which_method):
    """ The matrices are in bioinformatics format, so W is (genes, factors)
    and X is (genes, patients)"""
    g_w, k = W.shape
    g_x, n = X.shape
    assert g_w == g_x
    if which_method == 'NMF':
        # Enforse +ve results.
        # nnls() does not provide the convenience of accepting a matrix b argument
        H = np.array([nnls(W, b)[0] for b in X.T]).T
    else:
        H = lstsq(W, X)[0]

    assert H.shape == (k, n)
    return H


# noinspection PyPep8Naming
def make_survival_df(basename, facto_prefix, nc, component_list=None):
    """ Create a dataframe with patients in rows, RNAID as index, columns for the H components"""

    if component_list is None:
        component_list = range(1, nc + 1)  # 1-based naming of components
    meta_fname = '../Data/%s/%s_Metadata.tsv' % (basename, basename)
    meta_df = pd.read_csv(meta_fname, sep='\t', index_col='RNAID')
    rows_from_meta = meta_df.index.values
    x_fname = '../Data/%s/%s_PrunedExpression.tsv' % (basename, basename)
    df_x = pd.read_csv(x_fname, sep='\t', index_col=0)
    columns_from_X = list(df_x.columns)
    assert len(columns_from_X) > 1
    assert set(rows_from_meta) == set(columns_from_X)

    # Ensure the metadata is in same order as the columns from X
    meta_df = meta_df.reindex(columns_from_X)

    # Now pull out the X and W matrices and compute H from them
    X = np.asarray(df_x)
    df_w = FactorClustering.read_median_metagenes('TCGA_OV_VST', facto_prefix, nc)
    assert list(df_w.index.values) == list(df_x.index.values)

    W = np.asarray(df_w)
    H = compute_H_from_W_and_X(W, X, facto_prefix)

    # plt.hist(H.T, bins=40)
    # plt.show()
    assert H.shape == (nc, X.shape[1])

    # Copy the metagenes into the metadata
    n = H.shape[1]  # number of patients
    for k in component_list:
        colname = '%s_%d_of_%d' % (facto_prefix, k, nc)
        assert len(meta_df) == n
        meta_df[colname] = abs(H[k - 1, :])

        # Just to see what random looks like!
        # meta_df[colname] = np.random.randn(n)

    assert list(meta_df.index.values) == columns_from_X

    # Clean-up: remove nans and recode alive/dead

    df_clean = meta_df.drop(columns=['CaseID'])
    df_clean = df_clean.drop(columns=['age_at_diagnosis'])
    df_clean = df_clean.dropna()
    n_lost = len(meta_df) - len(df_clean)
    print("Number of subjects lost: ", n_lost)
    assert n_lost / len(meta_df) < 0.1  # No more than 10% lost
    status_dict = {'Alive': 0, 'Dead': 1}
    df_clean = df_clean.replace({'vital_status': status_dict})

    return df_clean


def plot_unstratified_survival(df, show=False):
    assert df.index.name == 'RNAID'
    assert 'survival_time' in df.columns
    assert 'vital_status' in df.columns

    kmf = KaplanMeierFitter()
    kmf.fit(df['survival_time'], df['vital_status'])

    kmf.plot(label='Unstratified survival')
    if show:
        plt.show()


def run_coxs_proportional_hazards(df):
    assert df.index.name == 'RNAID'
    assert 'survival_time' in df.columns
    assert 'vital_status' in df.columns

    cph = CoxPHFitter()
    cph.fit(df, duration_col='survival_time', event_col='vital_status')
    cph.print_summary(decimals=3)


if __name__ == '__main__':
    survival_df = make_survival_df('TCGA_OV_VST', 'ICA', 5, [2])
    run_coxs_proportional_hazards(survival_df)
