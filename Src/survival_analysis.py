""" Survival Analysis
   * Use JoinMetasamplesMetadata to:
      * Read training factors file (metagenes, W)
      * Read the training expression matrix
      * Compute H matrix by least squares
      * Read training survival data from metadata
   * Make Kaplan-Meier plots for each binarised factor
   * Perform Cox's Hazard with H factors as predictors, to hazard ratios and p-values
   * Bar plot summarising multiple experiments
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter

from join_metasamples_metadata import JoinMetasamplesMetadata  # used in main()


class SurvivalAnalysis:

    def __init__(self, train_basename, eval_basename, survival_or_relapse='os',
                 saveplots=False):
        """
        The constructor takes names of datasets, but does not access any data!
        The names are provided just so that the class can figure out what columns
        to use and how to name saved .pdfs.  Actual data is passed into the the plot
        methods as dataframes, whcih the client (see main()) will get from JoinMetasamplesMetadata.
        """

        tcga = 'TCGA_OV_VST'
        aocs = 'AOCS_Protein'
        both = 'BOTH_AOCS_TCGA'

        self.survival_or_relapse = survival_or_relapse

        assert train_basename in [tcga, aocs, both]
        assert eval_basename in [tcga, aocs, both]
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
            self.time_colname = 'survival_time'
            self.event_colname = 'vital_status'
            self.event_recode_dict = {'Alive': 0, 'Dead': 1}
        else:
            # AOCS or BOTH
            if survival_or_relapse == 'os':
                self.time_colname = 'donor_survival_time'
                self.event_colname = 'os_event'
            else:
                assert self.eval_basename == aocs
                self.time_colname = 'donor_relapse_interval'
                self.event_colname = 'pfs_event'
            self.event_recode_dict = {0: 0, 1: 1}  # no-op!

    @staticmethod
    def component_columns(df):
        """ Return a list of the factor columns (e.g. NMF_1_of_3) present in the df"""
        return [col for col in df.columns if col[:4] in ['NMF_', 'ICA_', 'PCA_']]

    def cleanup_and_threshold_components_df(self, df, components=None, percentile=50):
        # Clean-up: keep only the columns we need

        df_clean = df.dropna()
        n_lost = len(df) - len(df_clean)
        print("Number of subjects lost due to nan: ", n_lost)
        assert (n_lost <= 1) or (n_lost / len(df) < 0.1)  # No more than 10% lost
        df_clean = df_clean.replace({self.event_colname: self.event_recode_dict})

        # Threshold components (factors) to binarize
        t_df = df_clean.copy()
        components = components if components else self.component_columns(df)
        for c in components:
            t = np.percentile(df[c], percentile)
            t_df[c] = [1 if v > t else 0 for v in t_df[c]]
        return t_df

    def plot_unstratified_survival(self, survival_df, show=True):
        assert self.time_colname in survival_df.columns
        assert self.event_colname in survival_df.columns

        kmf = KaplanMeierFitter()
        kmf.fit(survival_df[self.time_colname] / 365, survival_df[self.event_colname])

        kmf.plot(label='Unstratified survival')
        plt.xlabel('Years')
        if show:
            plt.show()

        if self.saveplots:
            figpath = self.plots_dir + 'kaplan_meier_%s_unstratified.pdf' % self.eval_basename[:4]
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')

    def plot_component_stratified_survival(self, survival_df, component_name, show=True):
        """ We assume that given df is already thresholded with components having values 0 or 1"""
        assert self.time_colname in survival_df.columns
        assert self.event_colname in survival_df.columns
        assert component_name in survival_df.columns
        assert survival_df[component_name].max() == 1  # check thresholding
        assert survival_df[component_name].min() == 0

        df_low = survival_df[survival_df[component_name] == 0]
        df_high = survival_df[survival_df[component_name] == 1]

        # Run Cox's to get hazard ratio and p-value
        cph = self.run_once_coxs_proportional_hazards(survival_df, [component_name])
        assert len(cph.hazard_ratios_) == 1
        hazard_ratio = cph.hazard_ratios_[0]
        # noinspection PyProtectedMember
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
            component_name, self.train_basename[:4], self.eval_basename[:4],))
        plt.legend()

        if show:
            plt.show()

    def run_once_coxs_proportional_hazards(self, survival_df, component_list):
        """ Run coxs for the given components, eg. ['ICA_2_of_3',..] """
        assert self.event_colname in survival_df.columns
        assert self.time_colname in survival_df.columns
        for c in component_list:
            assert c in survival_df.columns
            assert survival_df[c].max() == 1  # check thresholding
            assert survival_df[c].min() == 0

        required_columns = [self.time_colname, self.event_colname] + component_list
        sub_df = survival_df[required_columns]

        cph = CoxPHFitter()
        cph = cph.fit(sub_df, duration_col=self.time_colname, event_col=self.event_colname)

        return cph

    @staticmethod
    def plot_hr_bars(thedict, show=True):
        """ Argument is a dictionary of dictionaries
        {component_name: {analysis_name:(hr, pvalue)}, where component_name is e.g. 'PCA_1_of_3',
        analysis_name is one of 'TCGA->TCGA(OR)', 'TCGA->OACS(OR), 'TCGA->OACS(PFS)'
        """

        all_components = list(thedict.keys())
        first_component = all_components[0]
        all_analyses = thedict[first_component].keys()
        n_components = len(all_components)
        n_analyses = len(thedict[first_component].keys())
        assert n_analyses == 3
        print(n_components, n_analyses)
        coloursdict = {'NMF': u'#1f77b4', 'ICA': u'#ff7f0e', 'PCA': u'#2ca02c'}
        # these are from the standard matplotlib colour cycle
        fill_colour = coloursdict[first_component[:3]]
        edge_colours = ['r', 'g', 'b']  # Colour to distinguish each analysis

        x = np.arange(n_components)
        w = 0.25
        for ai, a in enumerate(all_analyses):
            hrs = np.zeros(n_components)
            pvals = np.zeros(n_components)

            for ci, c in enumerate(all_components):
                hrs[ci], pvals[ci] = thedict[c][a]
                assert hrs[ci] > 0
                assert pvals[ci] < 1
                hrs[ci] = np.log2(hrs[ci])
            print(x)
            print(hrs)
            bar = plt.bar(x + ai * w, hrs, width=w - 0.03, label=a,
                          color=fill_colour, edgecolor=edge_colours[ai], linewidth=3)
            plt.xticks(x + w, all_components)
            # add p-vals
            for rect, p in zip(bar, pvals):
                h = rect.get_height()
                xx = rect.get_x()
                if h > 0:
                    plt.text(xx, h + 0.01, 'p=' + ("%.3f" % p)[1:], ha='left', va='bottom', size=8)
                else:
                    plt.text(xx, h - 0.01, 'p=' + ("%.3f" % p)[1:], ha='left', va='top', size=8)
            plt.gca().margins(0.1)
        plt.plot([-w / 2, n_components - w * 1.5], [0, 0], c='k')
        plt.ylabel('$log_2$(HR)')

        plt.legend()
        if show:
            plt.show()

    def run_survival_analysis(self, df, components=None, show=True):
        survival_df = self.cleanup_and_threshold_components_df(df, components)

        all_components = components if components else self.component_columns(df)

        print(all_components)
        report_df = pd.DataFrame(columns=('Component', 'Concordance', 'HR', 'p-val'))
        report_df.set_index('Component', inplace=True)

        for c in all_components:
            cph = self.run_once_coxs_proportional_hazards(survival_df, [c])
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

        cph = self.run_once_coxs_proportional_hazards(survival_df, all_components)
        cph.print_summary(decimals=3)

        plt.figure(figsize=(16, 20))
        for i, c in enumerate(all_components):
            plt.subplot(4, 3, i + 1)
            self.plot_component_stratified_survival(survival_df, c, show=False)
        # assert len(all_components) == 11
        # We've plotted 11 graphs, so fill up the 12th with unstratified survival
        plt.subplot(4, 3, 12)
        self.plot_unstratified_survival(survival_df, show=False)

        if self.saveplots:
            figpath = '../Plots/SurvivalAnalysis/' + 'multiple_kaplan_meier_%s_%s_%s.pdf' % (
                self.survival_or_relapse, self.train_basename[:4], self.eval_basename[:4])
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')

        if show:
            plt.show()

        return report_df  # the df will be needed for creating the summary bar plot


def run_one(train_basename, eval_basename, survival_or_relapse='os', saveplots=False, show=True):
    # Construct a dataframe indexed by patients, with columns for
    # metasamples and metadata
    df = JoinMetasamplesMetadata(train_basename, eval_basename).make_joined_df()
    print(df)

    survival_df = SurvivalAnalysis(train_basename, eval_basename,
                                   survival_or_relapse=survival_or_relapse,
                                   saveplots=saveplots).run_survival_analysis(df, show=show)

    return survival_df


def gene_specific_main():
    """ Quick hack to generate survival plots for CD38! """
    gene_ensg = 'ENSG00000004468'  # CD38
    gene_symbol = 'CD38'
    aocs, tcga = 'AOCS_Protein', 'TCGA_OV_VST'
    def one_sa(eval_basename, survival_or_relapse='os'):

        df = JoinMetasamplesMetadata(tcga, eval_basename).make_joined_gene_specific_df(
            [gene_ensg])
        df.rename(columns={gene_ensg: gene_symbol}, inplace=True)

        sa = SurvivalAnalysis(tcga, eval_basename,
                              survival_or_relapse=survival_or_relapse,
                              saveplots=False)
        survival_df = sa.cleanup_and_threshold_components_df(df, [gene_symbol])
        return sa, survival_df

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    sa, survival_df = one_sa(tcga)
    sa.plot_component_stratified_survival(survival_df, gene_symbol, show=False)
    plt.title("Kaplan-Meier %s %s (%s)" % (gene_symbol, tcga[:4], 'os'))

    plt.subplot(1, 3, 2)
    sa, survival_df = one_sa('AOCS_Protein')
    sa.plot_component_stratified_survival(survival_df, gene_symbol, show=False)
    plt.title("Kaplan-Meier %s %s (%s)" % (gene_symbol, aocs[:4], 'os'))

    plt.subplot(1, 3, 3)
    sa, survival_df = one_sa('AOCS_Protein', survival_or_relapse='pfs')
    sa.plot_component_stratified_survival(survival_df, gene_symbol, show=False)
    plt.title("Kaplan-Meier %s %s (%s)" % (gene_symbol, aocs[:4]
                                           , 'pfs'))

    figpath = sa.plots_dir + 'gene_specific_%s_survival.pdf' % gene_symbol
    plt.savefig(figpath, bbox_inches='tight')

    plt.show()


def main():
    show = False
    saveplots = True

    TCGA_os_df = run_one('TCGA_OV_VST', 'TCGA_OV_VST', saveplots=saveplots, show=show)
    AOCS_os_df = run_one('TCGA_OV_VST', 'AOCS_Protein', saveplots=saveplots, show=show)
    AOCS_pfs_df = run_one('TCGA_OV_VST', 'AOCS_Protein', survival_or_relapse='pfs',
                          show=show, saveplots=saveplots)

    # Build a dictionary of all results to create a summary bar-plot

    thedict = {}
    for component in TCGA_os_df.index:
        adict = {}
        for df, analysis_name in zip([TCGA_os_df, AOCS_os_df, AOCS_pfs_df],
                                     ['TCGA(OS)', 'AOCS(OS)', 'AOCS(PFS)']):
            adict[analysis_name] = (df['HR'][component], df['p-val'][component])
        thedict[component] = adict

    print(thedict)

    for prefix in ['NMF', 'ICA', 'PCA']:
        partial_dict = {k: v for k, v in thedict.items() if k[:3] == prefix}
        plt.figure(figsize=(3 * len(partial_dict), 4))
        SurvivalAnalysis.plot_hr_bars(partial_dict, show=show)
        if saveplots:
            figpath = '../Plots/SurvivalAnalysis/survival_summary_barplots_%s.pdf' % prefix
            print("Saving figure to", figpath)
            plt.savefig(figpath, bbox_inches='tight')


if __name__ == '__main__':
    main()
    # gene_specific_main()
