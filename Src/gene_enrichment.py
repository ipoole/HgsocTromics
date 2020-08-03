# ## Gene enrichment analysis using GOATOOLS
import gzip
import os
import pickle

import Bio.UniProt.GOA as GOA
import mygene
import numpy as np
import pandas as pd
import wget
from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
import matplotlib.pyplot as plt
import seaborn as sns


# noinspection PyStringFormat,PyMethodMayBeStatic
class GeneEnrichment:
    # Analyse standard deviation of components
    def __init__(self, basename, method):
        self.basename = basename
        self.method = method
        self._gene_symbols = None
        self.cache_dir = '../Cache/%s/GeneEnrichment/' % self.basename
        self.plots_dir = '../Plots/%s/GeneEnrichment/' % self.basename
        self.gene_column_name = 'Gene_ID' if 'Canon' in self.basename else 'GeneENSG'
        self._go_enrichment_study = None  # will be lazily evaluated

        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        self.download_and_cache_resources()  # Download ontology and annotations, if necessary
        self._gene_ontology = obo_parser.GODag('../DownloadedResources/go-basic.obo')

    def download_and_cache_resources(self):
        download_directory = '../DownloadedResources/'
        os.makedirs(download_directory, exist_ok=True)

        url_list = [
            'http://geneontology.org/gene-associations/goa_human.gaf.gz',
            'http://purl.obolibrary.org/obo/go/go-basic.obo']

        for url in url_list:
            p = os.path.join(download_directory, os.path.basename(url))
            if not os.path.exists(p):
                print("Downloading resource from %s ..." % url)
                wget.download(url, out=download_directory)

    def gene_symbols(self):
        """ We need HUGO style readable gene symbols for all genes in our study """
        if self._gene_symbols is None:
            expression_filename = '../Data/%s/%s_PrunedExpression.tsv' % \
                                  (self.basename, self.basename)

            # Read in only the first 'Gene_ID' column of the expression matrix
            expression_df = pd.read_csv(expression_filename, sep='\t',
                                        usecols=[self.gene_column_name])
            all_gene_ids = expression_df[self.gene_column_name].tolist()
            if 'Canon' in self.basename:
                # For the Canon data it's actually very simple, since the expression
                # matrix already gives HUGO gene names
                # These are of the form, e.g. 'AKT1_210', the number after the '_' is a TempO-Seq
                # identifier which we need to strip out.
                self._gene_symbols = [temposeq.split('_')[0] for temposeq in all_gene_ids]
                assert 'APOE' in self._gene_symbols
            else:
                # For AOCS and TCGA gene ids are in ENSG format
                assert 'ENSG' in all_gene_ids[0]
                # We'll need a dictionary, which we'll compute first time then cache to file
                ensgDictFile = self.cache_dir + 'ensgDict.pkl'
                if not os.path.exists(ensgDictFile):
                    mg = mygene.MyGeneInfo()
                    ginfo = mg.querymany(all_gene_ids, scopes='ensembl.gene')
                    ensgDict = {}
                    for g in ginfo:
                        ensg = g['query']
                        del g['query']
                        ensgDict[ensg] = g

                    print("Writing dictionary to %s..." % ensgDictFile)
                    with open(ensgDictFile, 'wb') as f:
                        pickle.dump(ensgDict, f)
                    print("Done.")
                with open(ensgDictFile, 'rb') as f:
                    ensgDict = pickle.load(f)

                for (ensg, g) in ensgDict.items():
                    if 'symbol' not in g.keys():
                        g['symbol'] = ensg  # ensure lookup always succeeds

                self._gene_symbols = [ensgDict[ensg]['symbol'] if ensg in ensgDict else ensg
                                      for ensg in all_gene_ids]

        return self._gene_symbols

    def read_metagene_matrix(self, factor_name):
        filename = '../Factors/%s/%s' % (self.basename, factor_name)
        metagene_df = pd.read_csv(filename, sep='\t')
        metagene_df.set_index(self.gene_column_name, inplace=True)
        metagene_matrix = np.asarray(metagene_df)
        assert metagene_matrix.ndim == 2
        return metagene_matrix

    def investigate_rank_threshold(self, metagene_matrix):
        # Analyse standard deviation of components
        assert metagene_matrix.ndim == 2
        n_stddev = 3.0
        for ci in range(metagene_matrix.shape[1]):
            metagene = metagene_matrix[:, ci]
            selection = self.select_influential_genes(metagene)
            stddev = np.std(metagene)
            print("Component %d, SD=%4.2f, #genes outside %3.1f SDs=%d" % (
                ci, stddev, n_stddev, len(selection)))

    def select_influential_genes(self, metagene):
        assert metagene.ndim == 1
        n_stddev = 3.0
        influence = abs(metagene)
        stddev = np.std(metagene)
        mean = np.mean(metagene)
        min_ = np.min(metagene)
        threshold = n_stddev * stddev
        symbols = self.gene_symbols()
        assert len(symbols) == len(metagene)
        gixpairs = zip(symbols, influence)

        if min_ >= 0:
            # Looks like its MNF...
            selection = [symbol for (symbol, v) in gixpairs if v - mean > threshold]
        else:
            selection = [symbol for (symbol, v) in gixpairs if abs(v - mean) > threshold]

        return selection

    def _extract_gene_enrichment_results_one_component(self, ci, gea_result, prefix, gea):
        tmp_fname = self.cache_dir + 'gea_tmp.tsv'  # Beware if running in parallel!
        significance_column = 'p_%s' % self.method
        if len(gea_result) > 0:
            with open(tmp_fname, 'w') as f:
                gea.prt_tsv(f, gea_result)
            ge_df = pd.read_csv(tmp_fname, sep='\t')

            ge_df.rename(columns={'# GO': 'GO_ID'}, inplace=True)
            ge_df.set_index('GO_ID', inplace=True)
            ge_df.drop(columns=['NS', 'enrichment', 'p_uncorrected'], inplace=True)
            ge_df = ge_df[ge_df[significance_column] <= 0.01]
            ge_df.insert(0, 'comp', ci + 1)  # 1-based naming of components
            ge_df.insert(0, 'facto_nc', prefix)
            return ge_df
        else:
            return None

    def go_enrichment_study(self):
        if self._go_enrichment_study is None:

            # Load the human annotations
            c = 0
            with gzip.open('../DownloadedResources/goa_human.gaf.gz', 'rt') as gaf:
                funcs = {}
                for entry in GOA.gafiterator(gaf):
                    c += 1
                    uniprot_id = entry.pop('DB_Object_Symbol')
                    funcs[uniprot_id] = entry
            # Our population is the set of genes we are analysing
            population = self.gene_symbols()
            print("We have %d genes in our population" % len(population))
            # Build associations from functional annotations we got from the gaf file
            associations = {}
            for x in funcs:
                if x not in associations:
                    associations[x] = set()
                associations[x].add(str(funcs[x]['GO_ID']))
            self._go_enrichment_study = \
                GOEnrichmentStudy(population, associations, self._gene_ontology,
                                  propagate_counts=True,
                                  alpha=0.01,
                                  methods=[self.method])
        return self._go_enrichment_study

    def perform_gene_enrichment_analysis(self, metagene_matrix, prefix):
        """ Analyse all the components in the given matrix, producing a single .tsv file
        summarising the enrichment results"""
        goe = self.go_enrichment_study()  # lazily deal with the GO, associations, population...

        gea_results_df_by_component = []
        n_comps = metagene_matrix.shape[1]
        study_gene_dict = {}
        for ci in range(n_comps):
            metagene = metagene_matrix[:, ci]
            study_genes = self.select_influential_genes(metagene)
            comp_name = '%s_%d_of_%d' % (prefix[:3], ci + 1, n_comps)
            study_gene_dict[comp_name] = study_genes
            print('\n%s[%d]: %s...' % (prefix, ci, str(study_genes[:10])))
            gea_result = goe.run_study(study_genes)

            # Get results into a dataframe per component.
            ge_df = self._extract_gene_enrichment_results_one_component(
                ci, gea_result, prefix, goe)
            if ge_df is not None:
                gea_results_df_by_component += [ge_df]

        # Concatenate the per-component dataframes into a single one
        gea_all_sig_results_df = pd.concat(gea_results_df_by_component, axis=0)
        all_results_fname = self.cache_dir + '%s_gea_%s.tsv' % (prefix, self.method)
        gea_all_sig_results_df.to_csv(all_results_fname, sep='\t')

        return all_results_fname, gea_all_sig_results_df, study_gene_dict

    def filter_and_generate_results(self, combined_results_fname):
        """ This should be run after a combined results file has been generated
        (see main() below).  We filter the GO_IDs sensibly and generate graphical
        visualizations of the go terms for each component """

        assert self._gene_ontology is not None

        min_depth = 3
        min_study_count = 0
        max_go_ids_to_plot = 12

        df = pd.read_csv(combined_results_fname, sep='\t')

        df = df[(df['depth'] >= min_depth) & (df['study_count'] >= min_study_count)]

        for facto, nc in zip(['NMF_3', 'ICA_5', 'PCA_3'], [3, 5, 3]):
            for c in range(1, nc + 1):
                component_df = df[(df['facto_nc'] == facto) & (df['comp'] == c)]

                if len(component_df) == 0:
                    print("Nothing to plot for", facto, c)
                    continue

                # Some components identify many go_ids and we can't plot them all.
                # So take the top n ranked by number of contributing genes.
                component_df = component_df.sort_values('study_count', ascending=False)
                n_all_terms = len(component_df)
                if n_all_terms > max_go_ids_to_plot:
                    limiting_study_count = component_df['study_count'].iloc[max_go_ids_to_plot]
                    print('***', facto, c, "Limiting %d terms to %d, min study count %d" %
                          (n_all_terms, max_go_ids_to_plot, limiting_study_count))

                go_ids = list(component_df['GO_ID'])[:max_go_ids_to_plot]
                lineage_fname = self.plots_dir + 'go_lineage_%d_of_%s.png' % (c, facto)
                recs = [self._gene_ontology[t] for t in go_ids]
                self._gene_ontology.draw_lineage(
                    recs, lineage_img=lineage_fname, draw_children=False, dpi=300)

                # Write files for each component containing the enriched genes
                gene_set = set()
                for row in component_df['study_items']:
                    row_of_genes = row.split(', ')
                    gene_set = gene_set.union(set(row_of_genes))
                gene_set_fname = self.plots_dir + 'go_gene_set_%d_of_%s.txt' % (c, facto)
                with open(gene_set_fname, 'w') as f:
                    f.write(' '.join(list(gene_set)))

    @staticmethod
    def jaccard(aset, bset):
        return len(aset.intersection(bset)) / len(aset.union(bset))

    @staticmethod
    def analyse_study_gene_jaccard(genes_dict):
        """ From a dictionary of component_name:candidate gene list,
        return a square, symmetric array of Jaccard similarities between all pairs
        of gene lists"""
        all_comps = list(genes_dict.keys())
        m = len(all_comps)
        jaccard_array = np.zeros((m, m))
        for ai, a in enumerate(all_comps):
            a_set = set(genes_dict[a])
            for bi, b in enumerate(all_comps):
                b_set = set(genes_dict[b])
                jaccard_array[ai, bi] = GeneEnrichment.jaccard(a_set, b_set)
        return jaccard_array

    def plot_jaccard_heatmap(self, jaccard_table, comp_names, show=True):
        plt.figure(figsize=(8, 8))
        # mask = np.triu(np.ones_like(jaccard_table, dtype=np.bool))
        sns.heatmap(jaccard_table, mask=None, square=False, cmap=plt.get_cmap('Reds'),
                    xticklabels=comp_names, yticklabels=comp_names,
                    annot=True, fmt=".2f")
        figpath = self.plots_dir + 'component_jaccard_heatmap.pdf'
        print("Saving figure to", figpath)
        plt.savefig(figpath, bbox_inches='tight')
        if show:
            plt.show()


# noinspection PyUnreachableCode
def main():
    ge = GeneEnrichment('TCGA_OV_VST', method='fdr')

    def run_one(facto_name, nc):
        factor_fname = '%s_median_factor_%d.tsv' % (facto_name, nc)
        prefix = '%s_%d' % (facto_name, nc)
        metagenes = ge.read_metagene_matrix(factor_fname)
        _, df, study_genes_dict = ge.perform_gene_enrichment_analysis(metagenes, prefix)
        return df, study_genes_dict

    combined_fname = ge.cache_dir + 'combined_gea_%s.tsv' % ge.method
    genes_dict_fname = ge.cache_dir + 'all_components_genes_dict'
    if True:
        # These are the 11 components we have decided to analyse.   Result will be to create
        # a file in cache for each factorizer - e.g. 'NMF_3_gea_al.tsv'
        nmf_df, nmf_genes_dict = run_one('NMF', 3)
        ica_df, ica_genes_dict = run_one('ICA', 5)
        pca_df, pca_genes_dict = run_one('PCA', 3)

        # Concatenate all the results together into s single file
        combined_df = pd.concat([nmf_df, ica_df, pca_df], axis=0)

        # Create a single dictionary of study (candidate) genes, to analyse below
        all_components_genes_dict = {**nmf_genes_dict, **ica_genes_dict, **pca_genes_dict}
        with open(genes_dict_fname, 'wb') as f:
            pickle.dump(all_components_genes_dict, f)

        combined_df.to_csv(combined_fname, sep='\t')

        print("Writen final combined enrichment results to %s." % combined_fname)

    # Analyse and plot the study genes intersections
    with open(genes_dict_fname, 'rb') as f:
        all_components_genes_dict = pickle.load(f)
    jaccard_table = GeneEnrichment.analyse_study_gene_jaccard(all_components_genes_dict)

    ge.plot_jaccard_heatmap(jaccard_table, list(all_components_genes_dict.keys()))

    ge.filter_and_generate_results(combined_fname)


if __name__ == '__main__':
    main()
