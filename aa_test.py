# write your code here
import pandas as pd
import scipy.stats as st
import numpy as np
from statsmodels.stats.power import TTestIndPower, TTestPower
import matplotlib.pyplot as plt


class ABTest:

    def __init__(self, filename):
        self.data = pd.read_csv(filename)

    def remove_outliers(self):
        if 'order_value' in self.data.columns:
            max_ov = self.data['order_value'].quantile(0.99)
            max_sd = self.data['session_duration'].quantile(0.99)
            self.data = self.data.query(f"order_value < {max_ov} \
                    and session_duration < {max_sd}")


class HypothesisTests(ABTest):

    def __init__(self, filename):
        super().__init__(filename)
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


class PowerTest(ABTest):

    def __init__(self, filename, effect, power, alpha):
        super().__init__(filename)
        self.ncont = self.data['group'].value_counts()['Control']
        self.nexp = self.data['group'].value_counts()['Experimental']
        self.size = TTestIndPower().solve_power(effect, power=power, nobs1=None, ratio=1.0, alpha=alpha)

    def print(self):
        print(f"Sample size: {round(self.size / 100) * 100}\n")
        print(f"Control group: {self.ncont}")
        print(f"Experimental group: {self.nexp}")


class ExploratoryDataAnalysis(ABTest):

    def __init__(self, filename):
        super().__init__(filename)
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.data['day'] = self.data['date'].dt.day

    def print(self):
        # Bar plot
        month = self.data['date'].dt.month_name().unique()[0]
        (self.data.groupby(['group', 'day'])
           .size()
           .reset_index(name='count')
           .pivot(index='day', columns='group', values='count')
           .plot(kind='bar', xlabel=month, ylabel='Number of sessions')#
           )
        plt.show()

        for col_name in ['order_value', 'session_duration']:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True)
            self.data.hist(column=col_name, by='group', ax=ax)
            xlab = " ".join(col_name.split("_")).capitalize()
            fig.supylabel("Frequency")
            fig.supxlabel(xlab)
            plt.show()

        self.remove_outliers()
        print(f"Mean: {self.data['order_value'].mean():0.2f}")
        print(f"Standard deviation: {self.data['order_value'].std():0.2f}")
        print(f"Max: {self.data['order_value'].max():0.2f}")


class MannWhitneyTest(ABTest):

    def __init__(self, filename):
        super().__init__(filename)
        self.test = None
        self.remove_outliers()

    def do_test(self):
        wdf = self.data.pivot(columns='group', values='order_value')
        self.test = st.mannwhitneyu(x=wdf['Control'].dropna(), y=wdf['Experimental'].dropna())

    def print(self):
        self.do_test()
        (sign, reject, target_equal) = (">", "no", "yes") if self.test.pvalue > 0.05 else ("<=", "yes", "no")
        print("Mann-Whitney U test")
        print(f"U1 = {self.test.statistic:0.1f}, p-value {sign} 0.05")
        print(f"Reject null hypothesis: {reject}")
        print(f"Distributions are same: {target_equal}")


class LogTransformation(HypothesisTests):
    def __init__(self, filename):
        super().__init__(filename)
        self.remove_outliers()
        self.data['log_order_value'] = np.log(self.data['order_value'])
        self.long = self.data.copy()
        self.data = self.data.pivot(columns='group', values='log_order_value')

    def print(self):
        fig, ax = plt.subplots()
        (self.long['log_order_value']
         .hist(legend=True, grid=False, ax=ax))
        ax.set_xlabel("Log order value")
        ax.set_ylabel("Frequency")
        plt.show()

        super().print()


# ot = HypothesisTests('aa_test.csv')
# ot = PowerTest('ab_test.csv', 0.2, 0.8, 0.05)
# ot = ExploratoryDataAnalysis('ab_test.csv')
# ot = MannWhitneyTest('ab_test.csv')
ot = LogTransformation('ab_test.csv')
ot.print()
