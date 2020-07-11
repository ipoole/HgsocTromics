# Python code to create the above Kaplan Meier curve
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import pandas as pd

# Example Data
# durations = [5, 6, 6, 2.5, 4, 4]
# event_observed = [1, 0, 0, 1, 1, 1]
#
# # create a kmf object
# kmf = KaplanMeierFitter()
#
# # Fit the data into the model
# kmf.fit(durations, event_observed, label='Kaplan Meier Estimate')
#
# # Create an estimate
# kmf.plot(ci_show=False)
# plt.show()

# ci_show is meant for Confidence interval,
# since our data set is too tiny, thus i am not showing it.

tcga_metatata_file = '../Data/TCGA_OV_VST/TCGA_OV_VST_Metadata.tsv'
pd.read_csv(tcga_metatata_file, sep='\t')

