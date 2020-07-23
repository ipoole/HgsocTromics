
#You need to load these libraries
#You might need to install them first - which you can do easily from bioconductor in R using the following lines

if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

# BiocManager::install("biomaRt")
# BiocManager::install("clusterProfiler")
# BiocManager::install("org.Hs.eg.db")

library("clusterProfiler")
library("org.Hs.eg.db")
library("biomaRt")
library(DOSE)
library(enrichplot)

getKeyGenes<-function(full,IC){
    
    col<-full[,IC]
    three_sds<-3*sd(col)
    av<-mean(col)

    above_ind<-which(col > av+three_sds)
    if (length(above_ind)>0){
        above<-rownames(full)[above_ind]
        df.a<-data.frame(Genes=above,Direction="above",IC=colnames(full)[IC])
    }else{
        df.a<-data.frame()
    }
    
    below_ind<-which(col < av-three_sds)
    if (length(below_ind)>0){
        below<-rownames(full)[below_ind]
        df.b<-data.frame(Genes=below,Direction="below",IC=colnames(full)[IC])
    }else{
        df.b<-data.frame()
    }
    
    keygenes<-rbind(df.a,df.b)

    return(keygenes)

}

# ====== Main Program =========


setwd("/home/ipoole/Documents/gitrepos/HgsocTromics/RSrc")

datasetname = 'AOCS_Protein'
facto = 'ICA'
n_components = 3

factor_file_path = sprintf('../Factors/%s/%s_median_factor_%d.tsv', datasetname, facto, n_components)
results_file_path = sprintf('../Cache/%s/GeneEnrichment/%s_%d_kegg_ge_all.tsv', datasetname, facto, n_components)
#dir.create(dirname(factor_file_path))
#dir.create(dirname(results_file_path))
cat('Starting ...')

extract_hugo_population <- function(ensg_list){
    ensembl<-useMart("ensembl",dataset="hsapiens_gene_ensembl")
    
    # Load your metagene matrix with genes as rows and genenames as rownames, ICs as columns with column names "IC1","IC2"... or something similar
    ICs<-read.table(factor_file_path, header=T, row.names=1, sep="\t")
    
    # Get the list of ensembl IDs in our study
    all_ensembl<-rownames(ICs)
    
    # Create a dataframe of these IDs, with columns for entrezgene_id and symbol
    gene_anno_all<-getBM(attributes=c('ensembl_gene_id', 'entrezgene_id', 'hgnc_symbol'),filters='ensembl_gene_id',values=all_ensembl,mart=ensembl)
    
    # Extract a plane list of all the entrez gene IDs
    hugo_genes_all<-as.character(gene_anno_all[gene_anno_all[,'ensembl_gene_id'] %in% all_ensembl,'entrezgene_id'])
    
    return (hugo_genes_all)
}

run_enrichKEGG <- function(hugo_genes, hugo_genes_all){
    genes<-as.character(keygenes[is.na(keygenes[,'Genes'])==FALSE,1])
    
    # genes is now a simple list of entrez gene ids.  Need to annotate same as for gene_anno above
    gene_anno<-getBM(attributes=c('ensembl_gene_id', 'entrezgene_id', 'hgnc_symbol'), filters='ensembl_gene_id', values=genes, mart=ensembl)
    hugo_genes<-unique(as.character(gene_anno[gene_anno[,'ensembl_gene_id'] %in% genes, 'entrezgene_id']))
    
    # So we can now pass these genes to gene_enrichment function
    
    kegg_result <- clusterProfiler::enrichKEGG(
        gene          = hugo_genes,
        universe      = hugo_genes_all,
        organism         = 'hsa',
        pAdjustMethod = "BH",
        pvalueCutoff  = 0.05,
        qvalueCutoff  = 0.05)
    
    return (keg_result)
}

# Load your metagene matrix with genes as rows and genenames as rownames, ICs as columns with column names "IC1","IC2"... or something similar
ICs<-read.table(factor_file_path, header=T, row.names=1, sep="\t")

hugo_genes_all <-  extract_hugo_population(rownames(ICs))


# We'll deal with just the first of three components.  getKeyGenes() returns a dataframe with ''above''Direction' column ('above' or 'below')
kegg_results <- list()
                                        
for (ic in 1: n_components){
    keygenes<-getKeyGenes(ICs, ic)
    genes<-as.character(keygenes[is.na(keygenes[,'Genes'])==FALSE,1])
    
    # genes is now a simple list of entrez gene ids.  Need to annotate same as for gene_anno above
    gene_anno<-getBM(attributes=c('ensembl_gene_id', 'entrezgene_id', 'hgnc_symbol'), filters='ensembl_gene_id', values=genes, mart=ensembl)
    hugo_genes<-unique(as.character(gene_anno[gene_anno[,'ensembl_gene_id'] %in% genes, 'entrezgene_id']))
    
    # So we can now pass these genes to gene_enrichment function
    
    kegg_results[[ic]] <- clusterProfiler::enrichKEGG(
                    gene          = hugo_genes,
                    universe      = hugo_genes_all,
                    organism         = 'hsa',
                    pAdjustMethod = "BH",
                    pvalueCutoff  = 0.05,
                    qvalueCutoff  = 0.05)
    
    # ekegg_readable = setReadable(ekegg, 'org.Hs.eg.db', 'ENTREZID')
    
    cat(sprintf("KEGG Gene Enrichment Analysis Done for %s, component %d, %d resuts found.", factor_file_path, ic, length(ekegg)))
}

# Add columns for factorizer method and component then concatonate rows
kegg_dfs <- list()
for (ic in 1: n_components){
    readable_kegg = setReadable(kegg_results[[ic]], 'org.Hs.eg.db', 'ENTREZID')
    df = as.data.frame(readable_kegg)
    df['Component'] = ic-1
    df['Factorizer'] = facto
    kegg_dfs[[ic]] = df
}

kegg_all_components <- do.call("rbind", kegg_dfs)
write.table(format(kegg_all_components, digits=3), file=results_file_path, sep="\t", row.names = F)

# 
# # Now display the results in various forms
# barplot(ekegg, showCategory=20)
# 
# # Provide readable symbols
# ekegg_readable = setReadable(ekegg, 'org.Hs.eg.db', 'ENTREZID')
# 
# # Make a gene concept network
# enrichplot::cnetplot(ekegg_readable)
# 
# # Make a heatmap type plot
# heatplot(ekegg_readable)
# 
# enrichplot::emapplot(ekegg_readable, piscale=1.5, layout='kk')

print ("KEGG Gene Enrichment Analysis Done")


