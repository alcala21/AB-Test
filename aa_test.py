# write your code here
import pandas as pd
import scipy.stats as st


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


ht = HypothesisTests('aa_test.csv')
ht.print()