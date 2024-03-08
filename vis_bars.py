import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rc
import sys, math
import argparse


parser = argparse.ArgumentParser('Stats renderer', add_help=False)
parser.add_argument('--metric', default='rmse', type=str, help='select metric to plot from: [`rmse`,`mae`,`obo`,`obz`]')
parser.add_argument('--group', default='length', type=str, help='select grouping either counts or rep duration: [`count`,`length`]')
parser.add_argument('--bins', default=5, type=int, help='Number of bins for grouping')

args = parser.parse_args()
plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams.update({'font.size': 26})
rc('font',**{'family':'serif','serif':['Times']})
rc('text', usetex=True)

fig, ax = plt.subplots()
fig.set_size_inches(7, 7)
lines = []

# df = pd.read_csv('out.csv')
# df["rep_dur"] = [0 for _ in range(len(df.index))]
df = pd.read_csv('datasets/repcount/repcount_test.csv')
df["rep_dur"] = [0 for _ in range(len(df.index))]
df["mae"] = [0 for _ in range(len(df.index))]
df_labels = pd.read_csv('datasets/repcount/repcount_test.csv')


for i,row in df_labels.iterrows():
    starts = [row[f"L{k}"] for k in range(1,302,2) if row[f"L{k}"] == row[f"L{k}"]]
    ends = [row[f"L{k}"] for k in range(2,302,2) if row[f"L{k}"] == row[f"L{k}"]]
    gt_count = row['gt']
    pred_count = row['ESCounts']
    mae = abs(gt_count - pred_count)/(gt_count + 1e-1)
    
    ts = 0
    for s,e in zip(starts,ends):
        ts += e-s
    if len(starts) == 0:
        divr = 1
    else:
        divr = len(starts)
    ts /= divr
    # print(df['name'])

    # df.loc[df['name'] == row['name'].split('.')[0],'rep_dur'] = ts
    df.loc[df['name'] == row['name'],'rep_dur'] = ts
    df.loc[df['name'] == row['name'],'mae'] = mae

# print(df['rep_dur'])
if args.group == 'count' :
    df_binned = pd.qcut(df['gt'], q=args.bins)
    print(df_binned)
    bins = {}
    for i in range(df_binned.size):
        k = (int(df_binned[i].left),int(df_binned[i].right))
        if k not in bins.keys():
            bins[k] = {'mae':[df.iloc[i]['mae']],
                        'diff':[abs(int(df.iloc[i]['gt'])-int(df.iloc[i]['ESCounts']))], 
                        'count':1}
        else:
            bins[k]['mae'].append(df.iloc[i]['mae'])
            bins[k]['diff'].append(abs(int(df.iloc[i]['gt'])-int(df.iloc[i]['ESCounts'])))
            bins[k]['count']+=1
    
    bins = dict(sorted(bins.items(), key=lambda x: x[0][0]))

elif args.group == 'length' :
    df['rep_dur'] = df['rep_dur']/df['fps']
    df_binned, bns = pd.qcut(df['rep_dur'], q=args.bins, retbins=True)
    print(df_binned)
    # print(df['rep_dur'].argmax())
    
    labels = {}
    for i in range(df_binned.size):
        k = (int(df_binned[i].left),int(df_binned[i].right))
        if int(df_binned[i].left) == int(bns[0]):
            labels[k] = 'XS'
        elif int(df_binned[i].left) == int(bns[1]):
            labels[k] = 'S'
        elif int(df_binned[i].left) == int(bns[2]):
            labels[k] = 'M'
        elif int(df_binned[i].left) == int(bns[3]):
            labels[k] = 'L'
        else:
            labels[k] = 'XL'

    bins = {}
    for i in range(df_binned.size):
        k = (int(df_binned[i].left),int(df_binned[i].right))
        if labels[k] not in bins.keys():
            bins[labels[k]] = {'mae':[df.iloc[i]['mae']],
                        'diff':[abs(int(df.iloc[i]['gt_count'])-int(df.iloc[i]['pred_count']))], 
                        'count':1}
        else:
            bins[labels[k]]['mae'].append(df.iloc[i]['mae'])
            bins[labels[k]]['diff'].append(abs(int(df.iloc[i]['gt_count'])-int(df.iloc[i]['pred_count'])))
            bins[labels[k]]['count']+=1

    new_bins ={'XS':bins['XS'],'S':bins['S'],'M':bins['M'],'L':bins['L'],'XL':bins['XL']}
    bins = new_bins


ax.set_axisbelow(True)
ax.yaxis.grid(color='gray', linestyle='solid')

# MAE BAR
if args.metric == 'mae':
    y = [sum(v['mae'])/v['count'] for v in bins.values()]
    ys = [vi for v in bins.values() for vi in v['mae']]
    if args.group == 'count' :
        bars = ax.bar([f"{k[0]+1}-{k[1]}" for k in bins.keys()],y,color=['#fdc187'])
    else:
        bars = ax.bar([k for k in bins.keys()],y,color=['#fdc187'])
    ax.bar_label(bars,fmt='{:.3f}')

    print('Avg',sum(ys)/len(ys))
    ax.hlines(y=sum(ys)/len(ys),xmin=-.4,xmax=4.4,label=f"Avg: {sum(ys)/len(ys):.3f}",linestyle='--',color='black')
    if args.group == 'count' :
        ax.text(y=sum(ys)/len(ys)+0.01,x=3,s=f"Avg: {sum(ys)/len(ys):.3f}")
    else:
        ax.text(y=sum(ys)/len(ys)+0.01,x=0,s=f"Avg: {sum(ys)/len(ys):.3f}")
    yl = "MAE"


# RMSE BAR
if args.metric == 'rmse':
    y = [math.sqrt(sum([d**2 for d in v['diff']])/v['count']) for v in bins.values()]
    ys = [d**2 for v in bins.values() for d in v['diff']]
    if args.group == 'count' :
        bars = ax.bar([f"{k[0]+1}-{k[1]}" for k in bins.keys()],y,color=['#eb8887'])
    else:
        bars = ax.bar([k for k in bins.keys()],y,color=['#eb8887'])
    ax.bar_label(bars,fmt='{:.3f}')

    t = math.sqrt(sum(ys)/len(ys))
    print('Avg',t)
    ax.hlines(y=t,xmin=-.4,xmax=4.4,label=f"Avg: {t:.3f}",linestyle='--',color='black')
    if args.group == 'count' :
        ax.text(y=t+0.1,x=0,s=f"Avg: {t:.3f}")
    else:
        ax.text(y=t+0.1,x=3,s=f"Avg: {t:.3f}")
    yl = "RMSE"
    



# OBO BAR
if args.metric == 'obo':
    y = [sum(map(lambda x: 1 if x<=1 else 0, v['diff']))/v['count'] for v in bins.values()]
    ys = [a for v in bins.values() for a in map(lambda x: 1 if x<=1 else 0, v['diff']) ]
    if args.group == 'count' :
        bars =  ax.bar([f"{k[0]+1}-{k[1]}" for k in bins.keys()],y,color=['#96c8a8'])
    else:
        bars =  ax.bar([k for k in bins.keys()],y,color=['#96c8a8'])
    ax.bar_label(bars,fmt='{:.3f}')

    t = sum(ys)/len(ys)
    print('Avg',t)
    ax.hlines(y=t,xmin=-.4,xmax=4.4,label=f"Avg: {t:.3f}",linestyle='--',color='black')
    if args.group == 'count' :
        ax.text(y=t+0.01,x=3,s=f"Avg: {t:.3f}")
    else:
        ax.text(y=t+0.01,x=3,s=f"Avg: {t:.3f}")
    yl = "OBO"
    


# OBZ BAR
if args.metric == 'obz':
    y = [sum(map(lambda x: 1 if x==0.0 else 0, v['diff']))/v['count'] for v in bins.values()]
    ys = [a for v in bins.values() for a in map(lambda x: 1 if x==0 else 0, v['diff']) ]
    if args.group == 'count' :
        bars = ax.bar([f"{k[0]+1}-{k[1]}" for k in bins.keys()],y,color=['#b599c5'])
    else:
        bars = ax.bar([k for k in bins.keys()],y,color=['#b599c5'])
    ax.bar_label(bars,fmt='{:.3f}')

    t = sum(ys)/len(ys)
    print('Avg',t)
    ax.hlines(y=t,xmin=-.4,xmax=4.4,label=f"Avg: {t:.3f}",linestyle='--',color='black')
    if args.group == 'count' :
        ax.text(y=t+0.01,x=3,s=f"Avg: {t:.3f}")
    else:
        ax.text(y=t+0.01,x=3,s=f"Avg: {t:.3f}")
    yl = "OBZ"
    




if args.group == 'length' :
    ax.set_xlabel('Ground Truth Number of Repetitions')
else:
    ax.set_xlabel('Repetition Length')

ax.set_ylabel(yl)

# Move left and bottom spines outward by 10 points
ax.spines['left'].set_position(('outward', 10))
ax.spines['bottom'].set_position(('outward', 10))
# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.tight_layout()
plt.savefig(f"{args.metric}_{args.group}.pdf",dpi=300)