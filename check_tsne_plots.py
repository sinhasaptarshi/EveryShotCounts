import pandas as pd
import numpy as np
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly as pl
import plotly.express as px
import math


t = TSNE(n_components=2, learning_rate='auto', init='pca', random_state=0)
train_data = pd.read_csv('datasets/repcount/train_with_fps.csv')

classes = train_data['type'].values  ## finding the classes for the videos
classes[classes == 'squant'] = 'squat'
classes[classes == 'pull_up'] = 'pullups'
classes[classes == 'push_up'] = 'pushups'
classes[classes == 'jump_jack'] = 'jumpjacks'
classes[classes == 'bench_pressing'] = 'benchpressing'
classes[classes == 'front_raise'] = 'frontraise'

unique_classes, counts = np.unique(classes, return_counts=True)

videos = train_data['name'].values  ### video names
overall_video_distace = 0
distance_across_videos = 0

# examplar_tokens = os.listdir('exemplar_tokens/')
combined_tokens = np.zeros([0, 1568, 768])
combined_classes = np.zeros(0)
types = np.zeros(0)
for i, type in enumerate(unique_classes):
    # if type == 'others':
    #     continue
    data = videos[classes == type]
    np.random.shuffle(data)
    tokens = np.zeros([0, 1568, 768])
    within_video_distance = 0
    selected_examplars = np.zeros([0, 1568, 768])
    data_count = 0
    for d in data:
        
        file_name = d.replace('.mp4', '.npz')
        try:
            examplars = np.load(os.path.join('exemplar_tokens', file_name))['arr_0']
        except:
            print(f"{file_name} does not exist")
            continue
        examplars = examplars.reshape(examplars.shape[0], examplars.shape[1], -1).transpose(0,2,1)
        np.random.shuffle(examplars)
        selected_examplars = np.concatenate([selected_examplars, examplars[:4]])
        if examplars.shape[0] >  1:
            data_count += 1
            
            # print((np.sqrt(((examplars[0] - examplars[1])**2).sum(1))).shape)

            distance = np.sqrt(((examplars[0] - examplars[1])**2).sum(1)).mean()
            within_video_distance += distance
        if data_count > 20:
            break


        # number_of_examplars = examplars.shape[0]
        # examplars = examplars.reshape(number_of_examplars, examplars.shape[1], -1).transpose(0,2,1)
        # np.random.shuffle(examplars)
        # tokens = np.concatenate([tokens, examplars[:3]])
        # if tokens.shape[0] > 20:
        #     print(tokens.shape)
        #     break
    # types = np.concatenate([types, i * np.ones(1000)])
    # tokens = tokens.reshape(-1, tokens.shape[-1])
    # np.random.shuffle(tokens)
    # combined_tokens = np.concatenate([combined_tokens, tokens[:1000]])
    print(f"within video distance for class {type} is {within_video_distance/data_count}")
    overall_video_distace += within_video_distance/data_count
    np.random.shuffle(selected_examplars)
    pair1 = selected_examplars[0::2]
    pair2 = selected_examplars[1::2]
    pair1 = pair1[:pair2.shape[0]]
    distance_across_videos += np.sqrt(((pair1 - pair2)**2).sum(-1)).mean()
    # combined_tokens = combined_tokens.reshape(-1, combined_tokens.shape[-1])
    combined_tokens = np.concatenate([combined_tokens, selected_examplars[:30]])
    combined_classes = np.concatenate([combined_classes, i*np.ones(selected_examplars[:30].shape[0])])

print('distance between rep tokens from same video', overall_video_distace/len(unique_classes))
print('disrance between rep tokens from same class', distance_across_videos/(len(unique_classes)))

# combined_tokens = t.fit_transform(combined_tokens)
#ig = px.scatter(X_embedded[(i*codebook.shape[0]):((i+1)* codebook.shape[0])], x=0, y=1).update_traces(marker=dict(color=colors))
# fig = px.scatter(combined_tokens, x=0, y=1, color=types)
# fig.write_image('tsne_for_classwise_tokens.png')
np.random.shuffle(combined_tokens)
pair1 = combined_tokens[0::2]
pair2 = combined_tokens[1::2]
pair1 = pair1[:pair2.shape[0]]
distance = np.sqrt(((pair1 - pair2)**2).sum(-1)).mean()
print('difference between tokens from random videos', distance)


dist = 0
num_classes = np.arange(len(unique_classes) - 1)
for i in num_classes:
    left_over = num_classes.tolist()
    left_over.remove(i)
    select_other = np.random.choice(left_over)
    select_tokens_i = combined_tokens[combined_classes==i]
    select_tokens_j = combined_tokens[combined_classes==select_other]
    how_many = min(select_tokens_i.shape[0], select_tokens_j.shape[0])
    pair_1 = select_tokens_i[:how_many]
    pair_2 = select_tokens_j[:how_many]
    dist += np.sqrt(((pair1 - pair2)**2).sum(-1)).mean()

print('difference between tokens from videos from different class', dist/len(unique_classes))




print(combined_tokens.shape)





