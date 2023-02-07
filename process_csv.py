import numpy as np
import pandas as pd

df = pd.read_csv(f'./data/WADI/A1/test.csv', sep=",", header=None,  skiprows=1).fillna(0)
df.drop([0,1,2], axis=1, inplace=True)
df.to_csv(f'./data/WADI/A1/test.csv', index=0)