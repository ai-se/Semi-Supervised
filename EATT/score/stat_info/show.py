"""
Use command "python show.py 0.1/bugzilla acc" to show the score of different algorithms.
"""
import sys
import pandas as pd
import numpy as np
file = sys.argv[1]
alg_id = int(sys.argv[2])
alg_set=[['CBS', 'TLEL2', 'OneWay', 'CoForest', 'FTF', 'EATT'], ['EALR', 'LT', 'CCUM', 'EATT']]

data = pd.read_csv(file+'.csv', index_col=0)
data_sig = pd.read_csv(file+'_sig.csv', index_col=0)

#print(data)

data = data.dropna(axis=1,how='any')
data = data[alg_set[alg_id]]
data_sig = data_sig[alg_set[alg_id]]
print(data)
print(data_sig)
name = dict(bugzilla='BUG', columba='COL', jdt='JDT', mozilla='MOZ', platform='PLA', postgres='POS', average='AVG')

value = data.values
data_sig = data_sig.values
row_name = data.index

print('\n')
len_r, len_c = np.shape(data)
# for i in range(len_c):
# print(np.tile(value[:, len_c-1], (len_r, 1)))
# print(value[:-1]<=np.tile(np.reshape(value[:-1, len_c-1], (-1, 1)), (1, len_c)))
data_sig[value[:-1]<=np.tile(np.reshape(value[:-1, len_c-1], (-1, 1)), (1, len_c))] = '-'
#data_sig[value[:-1]>np.tile(np.reshape(value[:-1, len_c-1], (-1, 1)), (1, len_c))] = '-'

for i in range(len_r-1):
	li = [name[row_name[i]]]
	for j, val in enumerate(value[i, :]):
		if j != len_c-1:
			li.append("%.4f" % val + '-' + data_sig[i, j][0])
		else:
			li.append("%.4f" % val + '-' + data_sig[i, j][0] + '\\\\')
	print("& "+" & ".join(li))