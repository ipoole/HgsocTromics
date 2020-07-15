""" Survival Analysis
   * Read training factors file (metagenes, W)
   * Read the training expression matrix
   * Compute H matrix by least squares
   * Read training survival data from metadata
   * Perfrorm Cox's Hazard with H factors as predictors
"""

import numpy as np
from factor_clustering import FactorClustering
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.linalg import lstsq


# noinspection PyPep8Naming
def compute_H_from_W_and_X(W, X):
    """ The matrices are in bioinformatics format, so W is (genes, factors)
    and X is (genes, patients)"""
    g_w, k = W.shape
    g_x, n = X.shape
    assert g_w == g_x

    computed_H = lstsq(W, X)[0]

    assert computed_H.shape == (k, n)
    return computed_H


# noinspection PyPep8Naming
def make_H_dataframe(basename, facto_prefix, nc):
    """ Create a datframe with patients in rows, RNAID as index, and columns for the H components"""
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
    H = compute_H_from_W_and_X(W, X)

    # plt.hist(H.T)
    # plt.show()
    assert H.shape == (nc, X.shape[1])

    # Copy the metagenes into the metadata
    n = H.shape[1]  # number of patients
    for k in range(nc):
        colname = '%s_%d_of_%d' % (facto_prefix, k + 1, nc)
        assert len(meta_df) == n
        meta_df[colname] = abs(H[k, :])

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
