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

