import pandas as pd

train_orig = pd.read_csv('datasets/repcount/train_with_fps.csv')
train_created = pd.read_csv('datasets/repcount/train_balanced_new.csv')
train_created['cycle_start_id'] = 0

for i in range(len(train_created)):
    row = train_created.loc[i]
    name = row['name']
    print(i)
    # print(name)
    count = row['count']
    start = row['L1']
    j = train_orig.loc[train_orig['name'] == name].index
    row_ = train_orig.loc[j[0]]
    # print(row_['count'])
    # if count == row_['count']:
    #    print(count)
    #    print(row_['count'])
    #    continue
    for k in range(int(row_['count'])):
    #   print(start)
    #   print(row_['L' + str(2*k + 1)].values)
      if row_['L' + str(2*k + 1)] == start:
         print(k)
         train_created['cycle_start_id'][i] = k
         break
      
train_created.to_csv('datasets/repcount/train_balanced_new.csv')