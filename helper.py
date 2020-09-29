import numpy as np
import itertools
import pandas as pd
import pickle


def flatten(sentences):
    flatted = []
    for sentence in sentences:
        for word in sentence:
            flatted.append(word)
    return flatted

def melt(songs):
    melt = []
    for index, song in songs.iterrows():
        for verse in song['lyric']:
            melt.append({'genre': song['genre'], 'verse': verse})  
    return melt

# https://stackoverflow.com/questions/37793118/load-pretrained-glove-vectors-in-python
def loadGloveModel(File='glove.6B.100d.txt'):
    f = open(File,'r')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    return gloveModel


#Refactor
restored_df = pd.read_csv(r'./export_dataframe.csv', converters={'lyric': eval})
restored_df_opt = pd.read_csv(r'./export_dataframe_opt.csv', converters={'lyric': eval})

def get_fresh_copy(frac=1, opt=False):
    if(opt):
        copy_df = pickle.loads(pickle.dumps(restored_df_opt.sample(frac=frac, random_state=1)))
    else:
        copy_df = pickle.loads(pickle.dumps(restored_df.sample(frac=frac, random_state=1)))
        
    return copy_df

def get_fresh_flatted_copy(frac=1, opt=False):
    copy_df = get_fresh_copy(frac, opt)
    copy_df['lyric'] = copy_df['lyric'].transform(lambda x: flatten(x))
    return copy_df