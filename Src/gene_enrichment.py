# ## Gene enrichment analysis using GOATOOLS
import os

from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
import Bio.UniProt.GOA as GOA
import gzip
import numpy as np
import pandas as pd
import pickle
import wget
import mygene


# noinspection PyStringFormat,PyMethodMayBeStatic
class GeneEnrichment:
    # Analyse standard deviation of components
    def __init__(self, basename, prefix):
        self.basename = basename
        self.prefix = prefix  # prefix to results files
        self._gene_symbols = None
        self.cache_dir = '../Cache/%s/GeneEnrichment/' % self.basename
        self.gene_column_name = 'Gene_ID' if 'Canon' in self.basename else 'GeneENSG'
        os.makedirs(self.cache_dir, exist_ok=True)

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

    def ranked_genes_by_component(self, metagene_matrix):
        W = metagene_matrix
        ranked_genes_by_component = {}
        for ci in range(metagene_matrix.shape[1]):
            _genes = self.select_influential_genes(W[:, ci])
            ranked_genes_by_component[ci] = _genes
        return ranked_genes_by_component

    def _perform_gene_enrichment_analysis_one_component(self, ci, gea_results_by_component, gea):
        tsv_name = self.cache_dir + '%s_gea_C%d.tsv' % (self.prefix, ci)
        if len(gea_results_by_component[ci]) > 0:
            with open(tsv_name, 'w') as f:
                gea.prt_tsv(f, gea_results_by_component[ci])
            ge_df = pd.read_csv(tsv_name, sep='\t')

            ge_df.rename(columns={'# GO': 'GO_ID'}, inplace=True)
            ge_df.set_index('GO_ID', inplace=True)
            ge_df.drop(columns=['NS', 'enrichment', 'p_uncorrected'], inplace=True)
            if 'p_fdr' in ge_df.columns:
                ge_df = ge_df[ge_df['p_fdr'] <= 0.05]
            else:
                ge_df = ge_df[ge_df['p_bonferroni'] <= 0.05]
            ge_df['Component'] = ci
            return ge_df
        else:
            return None

    def perform_gene_enrichment_analysis(self, metagene_matrix, method='fdr'):
        # Load the Gene Ontology
        n_comps = metagene_matrix.shape[1]

        self.download_and_cache_resources()   # Download ontology and annotations, if necessary
        gene_ontology = obo_parser.GODag('../DownloadedResources/go-basic.obo')

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

        gea = GOEnrichmentStudy(population, associations, gene_ontology,
                                propagate_counts=True,
                                alpha=0.05,
                                methods=[method])
        gea_results_by_component = {}
        rankings = self.ranked_genes_by_component(metagene_matrix)
        for ci in range(n_comps):
            study_genes = rankings[ci]
            print('\nComp. %d: %s...' % (ci, str(study_genes[:10])))
            gea_results_by_component[ci] = gea.run_study(study_genes)

        # Get results into a dataframe per component.  Easiest way is to use routine to
        # write a .tsv file, then read back and filter

        gea_results_df_by_component = []
        for ci in range(n_comps):
            ge_df = self._perform_gene_enrichment_analysis_one_component(
                ci, gea_results_by_component, gea)
            if ge_df is not None:
                gea_results_df_by_component += [ge_df]

        # Merge the per-component dataframes into a single one
        gea_all_sig_results_df = pd.DataFrame()
        gea_all_sig_results_df = gea_all_sig_results_df.append(gea_results_df_by_component)

        gea_all_sig_results_df.to_csv(self.cache_dir + '%s_gea_all.tsv' % self.prefix, sep='\t')


# noinspection PyUnreachableCode
def main():
    ge = GeneEnrichment('TCGA_OV_VST', 'NMF_3')
    metagenes = ge.read_metagene_matrix('NMF_median_factor_3.tsv')
    ge.perform_gene_enrichment_analysis(metagenes, method='bonferroni')

    ge = GeneEnrichment('TCGA_OV_VST', 'ICA_3')
    metagenes = ge.read_metagene_matrix('ICA_median_factor_3.tsv')
    ge.perform_gene_enrichment_analysis(metagenes, method='bonferroni')

    ge = GeneEnrichment('TCGA_OV_VST', 'PCA_3')
    metagenes = ge.read_metagene_matrix('PCA_median_factor_3.tsv')
    ge.perform_gene_enrichment_analysis(metagenes, method='bonferroni')

    # Demonstrate on the Canon dataset
    ge = GeneEnrichment('Canon_N200', 'NMF_3')
    metagenes = ge.read_metagene_matrix('NMF_median_factor_3.tsv')
    ge.perform_gene_enrichment_analysis(metagenes, method='bonferroni')


if __name__ == '__main__':
    main()
