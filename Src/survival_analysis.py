""" Survival Analysis
   * Read training factors file (metagenes, W)
   * Read the training expression matrix
   * Compute H matrix by least squares
   * Read training survival data from metadata
   * Perform Cox's Hazard with H factors as predictors
"""

import numpy as np
import os
from factor_clustering import FactorClustering
import pandas as pd
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter
from scipy.linalg import lstsq
from scipy.optimize import nnls


class SurvivalAnalysis:
    # noinspection PyPep8Naming

    def __init__(self, train_basename, eval_basename, survival_or_relapse='os',
                 saveplots=False):
        tcga = 'TCGA_OV_VST'
        aocs = 'AOCS_Protein'
        self.survival_or_relapse = survival_or_relapse

        assert train_basename in [tcga, aocs]
        assert eval_basename in [tcga, aocs]
        assert survival_or_relapse in ['os', 'pfs']

        self.train_basename = train_basename
        self.eval_basename = eval_basename
        self.survival_or_relapse = survival_or_relapse
        self.plots_dir = '../Plots/SurvivalAnalysis/'
        self.saveplots = saveplots
        os.makedirs(self.plots_dir, exist_ok=True)

        if eval_basename == tcga:
            # TCGA
            assert self.survival_or_relapse == 'os'  # no relapse data not available for TCGA
            self.patient_id_colname = 'RNAID'
            self.time_colname = 'survival_time'
            self.event_colname = 'vital_status'
            self.event_recode_dict = {'Alive': 0, 'Dead': 1}
        else:
            # AOCS
            self.patient_id_colname = 'Sample'
            # noinspection PyUnreachableCode
            if survival_or_relapse == 'os':
                self.time_colname = 'donor_survival_time'
                self.event_colname = 'os_event'
            else:
                self.time_colname = 'donor_relapse_interval'
                self.event_colname = 'pfs_event'
            self.event_recode_dict = {0: 0, 1: 1}  # no-op!

    # noinspection PyMethodMayBeStatic,PyPep8Naming
    def compute_H_from_W_and_X(self, W, X, which_method):
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
    def make_survival_df(self, facto_prefix, nc, component_list=None):
        """ Create a dataframe with patients in rows, RNAID as index, columns for the
        H components"""

        if component_list is None:
            component_list = range(1, nc + 1)  # 1-based naming of components

        meta_fname = '../Data/%s/%s_Metadata.tsv' % (self.eval_basename, self.eval_basename)
        meta_df = pd.read_csv(meta_fname, sep='\t', index_col=self.patient_id_colname,
                              usecols=[self.patient_id_colname, self.time_colname,
                                       self.event_colname])
        rows_from_meta = meta_df.index.values
        x_fname = '../Data/%s/%s_PrunedExpression.tsv' % (self.eval_basename, self.eval_basename)
        df_x = pd.read_csv(x_fname, sep='\t', index_col=0)
        columns_from_X = list(df_x.columns)
        assert len(columns_from_X) > 1
        assert set(rows_from_meta) == set(columns_from_X)

        # Ensure the metadata is in same order as the columns from X
        meta_df = meta_df.reindex(columns_from_X)

        # Now pull out the X and W matrices and compute H from them
        X = np.asarray(df_x)
        df_w = FactorClustering.read_median_metagenes(self.train_basename, facto_prefix, nc)
        assert list(df_w.index.values) == list(df_x.index.values)

        W = np.asarray(df_w)
        H = self.compute_H_from_W_and_X(W, X, facto_prefix)

        # plt.hist(H.T, bins=40)
        # plt.show()
        assert H.shape == (nc, X.shape[1])

        # Copy the metagenes into the metadata
        n = H.shape[1]  # number of patients
        for k in component_list:
            colname = '%s_%d_of_%d' % (facto_prefix, k, nc)
            assert len(meta_df) == n
            meta_df[colname] = H[k - 1, :]

            # Just to see what random looks like!
            # meta_df[colname] = np.random.randn(n)

        assert list(meta_df.index.values) == columns_from_X

        # Clean-up: keep only the columns we need

        df_clean = meta_df.dropna()
        n_lost = len(meta_df) - len(df_clean)
        print("Number of subjects lost due to nan: ", n_lost)
        assert n_lost / len(meta_df) < 0.1  # No more than 10% lost
        df_clean = df_clean.replace({self.event_colname: self.event_recode_dict})

        return df_clean

    def plot_unstratified_survival(self, df, show=True):
        assert df.index.name == self.patient_id_colname
        assert self.time_colname in df.columns
        assert self.event_colname in df.columns

        kmf = KaplanMeierFitter()
        kmf.fit(df[self.time_colname] / 365, df[self.event_colname])

        kmf.plot(label='Unstratified survival')
        plt.xlabel('Years')
        if show:
            plt.show()

        if self.saveplots:
            figpath = self.plots_dir + 'kaplan_meier_%s_unstratified.pdf' % self.eval_basename[:4]
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')

    # noinspection PyProtectedMember
    def plot_component_stratified_survival(self, df, component_name, show=True):
        """ We assume that given df is already thresholded with components having values 0 or 1"""
        assert df.index.name == self.patient_id_colname
        assert self.time_colname in df.columns
        assert self.event_colname in df.columns
        assert component_name in df.columns
        assert df[component_name].max() == 1   # check thresholding
        assert df[component_name].min() == 0

        df_low = df[df[component_name] == 0]
        df_high = df[df[component_name] == 1]

        # Run Cox's to get hazard ratio and p-value
        cph = self.run_once_coxs_proportional_hazards(df, [component_name])
        assert len(cph.hazard_ratios_) == 1
        hazard_ratio = cph.hazard_ratios_[0]
        p_value = cph._compute_p_values()[0]

        kmf = KaplanMeierFitter()
        kmf.fit(df_low[self.time_colname] / 365, df_low[self.event_colname])
        kmf.plot(label='%s Low' % component_name)

        kmf.fit(df_high[self.time_colname] / 365, df_high[self.event_colname])
        kmf.plot(label='%s High' % component_name)
        plt.plot([], [], ' ', label="HR=%5.3f" % hazard_ratio)
        plt.plot([], [], ' ', label="p-val=%5.3f" % p_value)
        plt.xlabel('Years')
        plt.title("Kaplan-Meier %s; %s->%s" % (
            component_name, self.train_basename[:4], self.eval_basename[:4], ))
        plt.legend()

        if show:
            plt.show()

    def run_once_coxs_proportional_hazards(self, df, component_list):
        """ Run coxs for the given components, eg. ['ICA_2_of_3',..] """
        assert df.index.name == self.patient_id_colname
        assert self.event_colname in df.columns
        assert self.time_colname in df.columns
        for c in component_list:
            assert c in df.columns
            assert df[c].max() == 1  # check thresholding
            assert df[c].min() == 0

        required_columns = [self.time_colname, self.event_colname] + component_list
        sub_df = df[required_columns]
        assert sub_df.index.name == self.patient_id_colname

        cph = CoxPHFitter()
        cph = cph.fit(sub_df, duration_col=self.time_colname, event_col=self.event_colname)

        return cph

    def make_combined_survival_df(self):
        """ The puts together a dataframe indexed by patient (RNAID) with survival metatada
        and *selected* metagenes (components).  These are the metagenes selected based
        on the cluster analysis as being stable
        """
        NMF_df = self.make_survival_df('NMF', 3, [1, 2, 3])
        ICA_df = self.make_survival_df('ICA', 5, [1, 2, 3, 4, 5])
        PCA_df = self.make_survival_df('PCA', 3, [1, 2, 3])

        # Ensure we get only one cope of these fields when we horizontally concatenate
        ICA_df.drop(columns=[self.time_colname, self.event_colname], inplace=True)
        PCA_df.drop(columns=[self.time_colname, self.event_colname], inplace=True)

        survival_all_df = pd.concat([NMF_df, ICA_df, PCA_df], axis=1)

        return survival_all_df

    def threshold_components_df(self, df, percentile=50):
        all_components = [c for c in df.columns if c not in [self.time_colname, self.event_colname]]
        t_df = df.copy()
        for c in all_components:
            t = np.percentile(df[c], percentile)
            t_df[c] = [1 if v > t else 0 for v in df[c]]
        return t_df

    def run_survival_analysis(self, show=True):
        thresholded_survival_df = self.threshold_components_df(
            self.make_combined_survival_df(), 50)
        all_components = thresholded_survival_df.columns
        all_components = [c for c in all_components
                          if c not in [self.time_colname, self.event_colname]]
        print(all_components)
        report_df = pd.DataFrame(columns=('Component', 'Concordance', 'HR', 'p-val'))
        report_df.set_index('Component', inplace=True)
        for c in all_components:
            cph = self.run_once_coxs_proportional_hazards(thresholded_survival_df, [c])
            assert len(cph.hazard_ratios_) == 1
            hazard_ratio = cph.hazard_ratios_[0]
            # noinspection PyProtectedMember
            p_val = cph._compute_p_values()[0]
            concordance = cph.concordance_index_
            report_df.loc[c] = [concordance, hazard_ratio, p_val]

        report_fname = self.plots_dir + 'survival_analysis_table_%s_%s_%s.tex' % (
            self.survival_or_relapse, self.train_basename[:4], self.eval_basename[:4])
        with pd.option_context('display.float_format', '{:0.3f}'.format):
            # print(report_df)
            report_tex = report_df.to_latex()
            print("Writing LaTeX report to ")
        if self.saveplots:
            print("Saving table to %s", report_fname)
            with open(report_fname, 'w') as f:
                f.write(report_tex)

        cph = self.run_once_coxs_proportional_hazards(thresholded_survival_df, all_components)
        cph.print_summary(decimals=3)

        plt.figure(figsize=(16, 20))
        for i, c in enumerate(all_components):
            plt.subplot(4, 3, i+1)
            self.plot_component_stratified_survival(thresholded_survival_df, c, show=False)
        assert len(all_components) == 11
        # We've plotted 11 graphs, so fill up the 12th with unstratified survival
        plt.subplot(4, 3, 12)
        self.plot_unstratified_survival(thresholded_survival_df, show=False)

        if self.saveplots:
            figpath = self.plots_dir + 'multiple_kaplan_meier_%s_%s.pdf' % (
                self.train_basename[:4], self.eval_basename[:4])
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')

        if show:
            plt.show()


def main():
    sa = SurvivalAnalysis('TCGA_OV_VST', 'TCGA_OV_VST', saveplots=True)
    sa.run_survival_analysis()

    sa = SurvivalAnalysis('TCGA_OV_VST', 'AOCS_Protein', saveplots=True)
    sa.run_survival_analysis()

    sa = SurvivalAnalysis('TCGA_OV_VST', 'AOCS_Protein',
                          survival_or_relapse='pfs', saveplots=True)
    sa.run_survival_analysis()


if __name__ == '__main__':
    main()
