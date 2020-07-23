echo Creating conda envirnment named 'tromics'
conda create -n tromics python=3.6 numpy scikit-learn pandas matplotlib pillow jupyterlab seaborn pygraphviz
source activate tromics
pip install mygene qgrid nimfa goatools biopython lifelines fisher nose
echo tromics environment created.  Don't forget to source activate tromics

