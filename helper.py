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