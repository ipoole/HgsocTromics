# Survival analysis tutorial

# install.packages("survival")
# install.packages("survminer")

library("survival")
library("survminer")

setwd('/home/ipoole/Documents/gitrepos/HgsocTromics/RSrc')

data("lung")
head(lung)

fit <- survival::survfit(Surv(time, status) ~ sex, data = lung)
print(fit)
summary(fit)
summary(fit)$table
ggsurvplot(fit, pval=T, conf.int=T, rist.table=T,
           # risk.table.col = "strata", linetype="strata", 
           surv.median.line = "hv", ggtheme = theme_bw(), palette = c("#E7B800", "#2E9FDF"))

ggsurvplot(
  fit,                     # survfit object with calculated statistics.
  pval = TRUE,             # show p-value of log-rank test.
  conf.int = TRUE,         # show confidence intervals for 
  # point estimaes of survival curves.
  conf.int.style = "step",  # customize style of confidence intervals
  xlab = "Time in days",   # customize X axis label.
  break.time.by = 200,     # break X axis in time intervals by 200.
  ggtheme = theme_light(), # customize plot and risk table with a theme.
  risk.table = "abs_pct",  # absolute number and percentage at risk.
  risk.table.y.text.col = T,# colour risk table text annotations.
  risk.table.y.text = FALSE,# show bars instead of names in text annotations
  # in legend of risk table.
  ncensor.plot = TRUE,      # plot the number of censored subjects at time t
  surv.median.line = "hv",  # add the median survival pointer.
  legend.labs = 
    c("Male", "Female"),    # change legend labels.
  palette = 
    c("#E7B800", "#2E9FDF") # custom color palettes.
)
lung

tcga_file = '../Data/TCGA_OV_VST/TCGA_OV_VST_Metadata.tsv'
df = read.table(tcga_file, header=T, row.names=1, sep="\t")
head(df)

decode <- function(s){
  if (s == 'Alive'){
    return(1)
  }
  else{
    return(2)
  }
}

df[,'vital_status'] <- sapply(df[,'vital_status'], decode)
df = na.omit(df)
df$survival_time = df[,'survival_time']
df$vital_status = df[,'vital_status']

fit <- survival::survfit(Surv('survival_time', 'vital_status'), data=df)

