import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import numpy as np

data = pd.read_csv('data_2009_updated2.csv')

#print(data.head(1))
print(data.columns.values)

# Case 1

gdp_model = ols("GDP_o ~ IPTOT_o", data=data).fit()
gdp_model_summary = gdp_model.summary()

print(gdp_model_summary)

fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(gdp_model, "IPTOT_o", fig=fig)
fig.savefig('gdp_model-IPTOT_o.png', dpi=600)

#x = data[['IPTOT_o']] # predictor
#y = data[['GDP_o']] # dependent
#_, confidence_interval_lower, confidence_interval_upper = wls_prediction_std(gdp_model)
#fig, ax = plt.subplots(figsize=(10,7))
#ax.plot(x, y, 'o', label="data")
#ax.plot(x, gdp_model.fittedvalues, 'g--.', label="OLS")
#ax.plot(x, confidence_interval_upper, 'r--')
#ax.plot(x, confidence_interval_lower, 'r--')
#ax.legend(loc='best');
#fig.savefig('gdp_model-IPTOT_o-trend.png', dpi=600)


# Case 2

gdp_model = ols("GDP_o ~ IPTOT_o + XPTOT_o", data=data).fit()
gdp_model_summary = gdp_model.summary()

print(gdp_model_summary)

fig = plt.figure(figsize=(20,12))
fig = sm.graphics.plot_partregress_grid(gdp_model, fig=fig)
fig.savefig('gdp_model-IPTOT_o+XPTOT_o.png', dpi=600)







