import numpy as np
import pandas as pd
import math

data = pd.read_csv('datasets/repcount/validtest_with_fps.csv')
columns = data.columns
for i in range(151):
    data[f'M{i+1}'] = np.nan
for i in range(len(data)):
    row = data.iloc[i]
    for j in range(151):
        start = row[f'L{2*j+1}']
        end = row[f'L{2*j+2}']
        if not math.isnan(start) and not math.isnan(end) and start != end:
            frame_num = np.random.randint(1, end - start)
            mid = start + frame_num
            data[f'M{j+1}'][i] = mid

data.to_csv('datasets/repcount/validtest_with_fps.csv')