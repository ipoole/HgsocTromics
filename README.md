# HgsocTromics
Code for my MSc Dissertation on Patterns of gene expression in high grade serous ovarian cancer (HGSOC)

Tromics: Transcriptomics!

## Setting up a Python3 environment

```shell script
$ conda create -n tromics python=3.6 numpy scikit-learn pandas matplotlib pillow jupyterlab seaborn
$ source activate tromics
$ pip install mygene qgrid nimfa goatools biopython nose
```
The above has been captuured in ```create_tromics_conda_invironment.sh``` so you can just:
```shell script
$ cd HgsocTromics
$ ./create_tromics_conda_invironment
```

## Steps to checkout and build

First setup a Python 3.6 environment withthe above packages.

Then:
```shell script
$ cd ~/Documents/gitrepos
$ git clone git@github.com:ipoole/HgsocTromics.git
$ ls HgsocTromics
Cache Notebooks README.md  RSrc  Src

$ cd HgsocTromics/Src
$ nosetests
..................................................
----------------------------------------------------------------------
Ran 50 tests in 12.040s

OK
```
Unit tests are based on trivial ('Mini') expression datasets
in ...Data/Mini_AOCA and .../Data/Mini_Canon, which are committed.

Note that on the first run of unit tests there will be warnings about
factorization algorithms failing to converge (due to 'Mini' datasets).   The second
run of ``` nosetests ``` will be clean as above since the factorizations
are cached.

## Configuring PyCharm

I have attempted to commit the PyCharm recommended .idea files, so this *should*
work out of the box (except for the python interpreter).  If not, a few things to check:
* Specify the correct Python 3.6 interpreter with the above packages installed.
* Unit test run templates to use ```nosetests``` and working directory .../Src

You should be able to run init tests from within PyCharm

## Adding real data

The substantive data is *not* committed to git and must be obtained seperately.
There are currently three datasets (in addition to the two 'Mini' datasets): 
AOCS, TCGA and Canon_N200.  These are added into the .../Data folder, which will then
look like:
```shell script
$ ls -r *
TCGA_OV_VST:
TCGA_OV_VST_Metadata.tsv  TCGA_OV_VST_Expression.tsv

Mini_Canon:
Mini_Canon_Expression.tsv

Mini_AOCS:
Mini_AOCS_Expression.tsv

Canon_N200:
Canon_N200_Expression.tsv

AOCS_Protein:
AOCS_TPM_VST.csv        AOCS_Protein_Scraped_Metadata.csv  AOCS_Protein_Expression.tsv
aocs_raw_figure_e6.txt  AOCS_Protein_Metadata.tsv

```

## Running on real data

.... TODO
