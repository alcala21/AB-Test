# write your code here
import pandas as pd
import scipy.stats as st
import numpy as np
from statsmodels.stats.power import TTestIndPower, TTestPower
import matplotlib.pyplot as plt


class HypothesisTests:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)

        self.levene = st.levene(self.data.iloc[:, 0].values, self.data.iloc[:, 1].values, center='mean')
        self.equal_var = self.levene.pvalue > 0.05

        self.t = st.ttest_ind(self.data.iloc[:, 0].values, self.data.iloc[:, 1].values, equal_var=self.equal_var)
        self.equal_mean = self.t.pvalue > 0.05

    def print(self):
        self.print_test("Levene's Test", self.equal_var, "W", self.levene.statistic, "Variances")
        print()
        self.print_test("T-test", self.equal_mean, "t", self.t.statistic, "Means")

    def print_test(self, test_name, criterion, stat_name, stat_value, targets):
        (sign, reject, target_equal) = (">", "no", "yes") if criterion else ("<=", "yes", "no")
        print(test_name)
        print(f"{stat_name} = {stat_value:0.3f}, p-value {sign} 0.05")
        print(f"Reject null hypothesis: {reject}")
        print(f"{targets} are equal: {target_equal}")


class PowerTest:
    def __init__(self, effect, power, alpha, filename):
        self.data = pd.read_csv(filename)
        self.ncont = self.data['group'].value_counts()['Control']
        self.nexp = self.data['group'].value_counts()['Experimental']

        analysis = TTestIndPower()
        self.size = analysis.solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)

    def print(self):
        print(f"Sample size: {round(self.size / 100) * 100}\n")
        print(f"Control group: {self.ncont}")
        print(f"Experimental group: {self.nexp}")


# EDA
df = pd.read_csv('ab_test.csv')
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
month = df['date'].dt.month_name().unique()[0]
dff = (df.groupby(['group', 'day'])
       .size()
       .reset_index(name='count')
       .pivot(index='day', columns='group', values='count')
       )
dff.plot(kind='bar', xlabel=month, ylabel='Number of sessions')#
plt.show()

# Histograms
for col_name in ['order_value', 'session_duration']:
    fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
    ax = df.hist(column=col_name, by='group', ax=ax)
    xlab = " ".join(col_name.split("_")).capitalize()
    fig.supylabel("Frequency")
    fig.supxlabel(xlab)
    plt.show()

# Statistics
max_ov = df['order_value'].quantile(0.99)
max_sd = df['session_duration'].quantile(0.99)

ov = (df.query(f"order_value < {max_ov} \
        and session_duration < {max_sd}")
        ['order_value'].values)

print(f"Mean: {ov.mean():0.02f}")
print(f"Standard deviation: {ov.std():0.02f}")
print(f"Max: {ov.max():0.02f}")
