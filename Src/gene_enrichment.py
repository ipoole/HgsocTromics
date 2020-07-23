# ## Gene enrichment analysis using GOATOOLS
import os

from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
import Bio.UniProt.GOA as GOA
import gzip
import numpy as np
import pandas as pd
import pickle
import tempfile
import wget
import mygene


# noinspection PyStringFormat,PyMethodMayBeStatic
class GeneEnrichment:
    # Analyse standard deviation of components
    def __init__(self, basename, method):
        self.basename = basename
        self.method = method
        self._gene_symbols = None
        self.cache_dir = '../Cache/%s/GeneEnrichment/' % self.basename
        self.gene_column_name = 'Gene_ID' if 'Canon' in self.basename else 'GeneENSG'
        self._go_enrichment_study = None     # will be lazily evaluated
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
            ge_df = ge_df[ge_df[significance_column] <= 0.05]
            ge_df['Component'] = ci
            return ge_df
        else:
            return None

    def go_enrichment_study(self):
        if self._go_enrichment_study is None:
            # Load the Gene Ontology
            self.download_and_cache_resources()  # Download ontology and annotations, if necessary
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
            self._go_enrichment_study = \
                GOEnrichmentStudy(population, associations, gene_ontology,
                                  propagate_counts=True,
                                  alpha=0.05,
                                  methods=[self.method])
        return self._go_enrichment_study

    def perform_gene_enrichment_analysis(self, metagene_matrix, prefix):
        """ Analyse all the components in the given matrix, producing a single .tsv file
        summarising the enrichment results"""
        goe = self.go_enrichment_study()    # lazily deal with the GO, associations, population...

        gea_results_df_by_component = []
        n_comps = metagene_matrix.shape[1]
        for ci in range(n_comps):
            metagene = metagene_matrix[:, ci]
            study_genes = self.select_influential_genes(metagene)
            print('\n%s[%d]: %s...' % (prefix, ci, str(study_genes[:10])))
            gea_result = goe.run_study(study_genes)

            # Get results into a dataframe per component.
            ge_df = self._extract_gene_enrichment_results_one_component(
                ci, gea_result, prefix, goe)
            if ge_df is not None:
                gea_results_df_by_component += [ge_df]

        # Concatenate the per-component dataframes into a single one
        gea_all_sig_results_df = pd.concat(gea_results_df_by_component, axis=0)
        all_results_fname = self.cache_dir + '%s_gea_%s_all.tsv' % (prefix, self.method)
        gea_all_sig_results_df.to_csv(all_results_fname, sep='\t')


# noinspection PyUnreachableCode
def main():
    ge = GeneEnrichment('TCGA_OV_VST', method='bonferroni')

    def run_one(facto_name, nc):
        factor_fname = '%s_median_factor_%d.tsv' % (facto_name, nc)
        prefix = '%s_%d' % (facto_name, nc)
        metagenes = ge.read_metagene_matrix(factor_fname)
        ge.perform_gene_enrichment_analysis(metagenes, prefix)

    # These are the 11 components we have decided to analyse.   Result will be to create
    # a file in cache for each factorizer - e.g. 'NMF_3_gea_al.tsv'
    run_one('NMF', 3)
    run_one('ICA', 5)
    run_one('PCA', 3)


if __name__ == '__main__':
    main()
