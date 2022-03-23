# write your code here
import pandas as pd
import scipy.stats as st
import numpy as np
from statsmodels.stats.power import TTestIndPower, TTestPower
import matplotlib.pyplot as plt


class HypothesisTests:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.levene = None
        self.equal_var = None
        self.t = None
        self.equal_mean = None

    def do_test(self):
        self.levene = st.levene(self.data.iloc[:, 0].dropna().values, self.data.iloc[:, 1].dropna().values)
        self.equal_var = self.levene.pvalue > 0.05

        self.t = st.ttest_ind(self.data.iloc[:, 0].dropna().values, self.data.iloc[:, 1].dropna().values, equal_var=self.equal_var)
        self.equal_mean = self.t.pvalue > 0.05

    def print(self):
        self.do_test()
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


class ExploratoryDataAnalysis:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['day'] = self.data['date'].dt.day

    def plots(self):
        # Bar plot
        month = self.data['date'].dt.month_name().unique()[0]
        (self.data.groupby(['group', 'day'])
           .size()
           .reset_index(name='count')
           .pivot(index='day', columns='group', values='count')
           .plot(kind='bar', xlabel=month, ylabel='Number of sessions')#
           )
        plt.show()

        # Histograms
        for col_name in ['order_value', 'session_duration']:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            self.data.hist(column=col_name, by='group', ax=ax)
            xlab = " ".join(col_name.split("_")).capitalize()
            fig.supylabel("Frequency")
            fig.supxlabel(xlab)
            plt.show()

    def print(self):
        # Statistics
        max_ov = self.data['order_value'].quantile(0.99)
        max_sd = self.data['session_duration'].quantile(0.99)

        ov = (self.data.query(f"order_value < {max_ov} \
                and session_duration < {max_sd}")
                ['order_value'].values)

        print(f"Mean: {ov.mean():0.02f}")
        print(f"Standard deviation: {ov.std():0.02f}")
        print(f"Max: {ov.max():0.02f}")


class MannWhitneyTest:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.prepare_data()
        self.test = st.mannwhitneyu(x=self.data['Control'].dropna(), y=self.data['Experimental'].dropna())

    def prepare_data(self):
        max_ov = self.data['order_value'].quantile(0.99)
        max_sd = self.data['session_duration'].quantile(0.99)
        self.data = (self.data.query(f"order_value < {max_ov} \
                and session_duration < {max_sd}")
                .pivot(columns='group', values='order_value'))

    def print(self):
        (sign, reject, target_equal) = (">", "no", "yes") if self.test.pvalue > 0.05 else ("<=", "yes", "no")
        print("Mann-Whitney U test")
        print(f"U1 = {self.test.statistic:0.1f}, p-value {sign} 0.05")
        print(f"Reject null hypothesis: {reject}")
        print(f"Distributions are same: {target_equal}")


class LogTransformation(HypothesisTests):
    def __init__(self, filename):
        super().__init__(filename)
        self.prepare_data()
        self.data['log_order_value'] = np.log(self.data['order_value'])
        self.long = self.data.copy()
        self.data = self.data.pivot(columns='group', values='log_order_value')

    def prepare_data(self):
        max_ov = self.data['order_value'].quantile(0.99)
        max_sd = self.data['session_duration'].quantile(0.99)
        self.data = self.data.query(f"order_value < {max_ov} \
                and session_duration < {max_sd}")

    def plot(self):
        fig, ax = plt.subplots()
        (self.long['log_order_value']
         .hist(legend=True, grid=False, ax=ax))
        ax.set_xlabel("Log order value")
        ax.set_ylabel("Frequency")
        plt.show()


lt = LogTransformation('ab_test.csv')
lt.plot()
lt.print()
