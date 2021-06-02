import tushare as ts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


datapath1 = "./SH600519.csv"
if os.path.exists(datapath1):
    df = pd.read_csv(datapath1, header=0)
else:
    df = ts.get_k_data('600519', ktype='D', start='2010-04-26', end='2021-04-26')
    df.to_csv(datapath1)

data = np.array(df)
print(data.shape)
train_set, test_set = data[0:-300, 2], data[-300:, 2]
print(train_set.shape, test_set.shape)
