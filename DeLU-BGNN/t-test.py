import scipy.sparse as sp
import numpy as np




import numpy as np
from scipy import stats

mean1 = 60.2
mean2 = 59.9

std1 = 0.2
std2 = 0.2

nobs1 = 1000
nobs2 = 1000

# mean1 = 64.5
# mean2 = 59.2
#
# std1 = 0.4
# std2 = 0.4
#
# nobs1 = 1000
# nobs2 = 1000
#
modified_std1 = np.sqrt(np.float32(nobs1)/np.float32(nobs1-1)) * std1
modified_std2 = np.sqrt(np.float32(nobs2)/np.float32(nobs2-1)) * std2

(statistic, pvalue) = stats.ttest_ind_from_stats(mean1=mean1, std1=modified_std1, nobs1=10, mean2=mean2, std2=modified_std2, nobs2=10)

print("t statistic is: ", statistic)
print("pvalue is: ", pvalue)