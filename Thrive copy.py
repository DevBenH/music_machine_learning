import pandas as pd
import librosa
import librosa.display
import matplotlib.pylab as plt
import sklearn
import numpy as np
import os 

sample_music = pd.read_csv('/Users/BenH/Desktop/Thrive/Data/features_30_sec.csv')
#print(sample_music.tail(5))

y, sr = librosa.load('/Users/BenH/Desktop/Thrive/Data/genres_original/jazz/jazz.00004.wav')
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
#print('Tempo: {:2f}'.format(tempo))

spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
spec_time = range(len(spec_cent))
f = librosa.frames_to_time(spec_time)
librosa.display.waveshow(y, sr=sr, alpha=0.4)
plt.plot(f, sklearn.preprocessing.minmax_scale(spec_cent, axis=0), color='r')