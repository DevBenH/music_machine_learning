# Libraries
import IPython.display as ipd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
import pandas as pd
import librosa
import librosa.display
import matplotlib.pylab as plt
import sklearn
import numpy as np
import os 
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.linear_model import LinearRegression
import IPython.display as ipd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing
data = pd.read_csv('/Users/BenH/Desktop/Thrive/Data/features_30_sec.csv')
# Read data
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier, XGBRFClassifier
from xgboost import plot_tree, plot_importance

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
# Extract labels

import IPython.display as ipd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import preprocessing

#with open('/Users/BenH/Desktop/Thrive/Data/features_30_sec.csv','a') as fd:
    #fd.write('hello')
import statistics 
import itertools
#EXTRACT THE FEATURES FROM THE SONG

def extract_song_features(filepath):


    y, sr = librosa.load(filepath)

    #chroma_stft_mean
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    new_chroma = (sum(chroma) / len(chroma))
    chroma_mean = (sum(new_chroma) / len(new_chroma))

    flattened_chroma = itertools.chain.from_iterable(chroma)
    chroma_var = statistics.variance(flattened_chroma)

    rms = librosa.feature.rms(y=y)
    new_rms = (sum(rms)/len(rms))
    rms_mean = ((sum(new_rms)/len(new_rms)))

    flattened_rms = itertools.chain.from_iterable(rms)
    rms_var = (statistics.variance(flattened_rms))

    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    new_spec_cent = (sum(spec_cent)/len(spec_cent))
    spec_cent_mean = ((sum(new_spec_cent)/len(new_spec_cent)))

    flattened_spec_cent = itertools.chain.from_iterable(spec_cent)
    spec_cent_var = (statistics.variance(flattened_spec_cent))

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    new_spec_bw = (sum(spec_bw)/len(spec_bw))
    spec_bw_mean = ((sum(new_spec_bw)/len(new_spec_bw)))

    flattened_spec_bw = itertools.chain.from_iterable(spec_bw)
    spec_bw_var = (statistics.variance(flattened_spec_bw))

    #through trial and error I was able to find that they used a rol_percent of 0.85
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    new_rolloff = (sum(rolloff)/len(rolloff))
    rolloff_mean = ((sum(new_rolloff)/len(new_rolloff)))

    flattened_rolloff= itertools.chain.from_iterable(rolloff)
    rolloff_var = (statistics.variance(flattened_rolloff))

    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    new_zero_crossing_rate = (sum(zero_crossing_rate)/len(zero_crossing_rate))
    zero_crossing_rate_mean = ((sum(new_zero_crossing_rate)/len(new_zero_crossing_rate)))

    flattened_zero_crossing_rate= itertools.chain.from_iterable(zero_crossing_rate)
    zero_crossing_rate_var = (statistics.variance(flattened_zero_crossing_rate))

    #harmony_mean
    #CANNOT GET A NEAR ENOUGH VALUE FOR THIS

    #harmony_var

    #per_mean

    #per_var

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    mfccs1_mean = ((sum(mfccs[0])/len(mfccs[0])))
    mfccs1_var = (statistics.variance(mfccs[0]))

    mfccs2_mean = ((sum(mfccs[1])/len(mfccs[1])))
    mfccs2_var = (statistics.variance(mfccs[1]))

    mfccs3_mean = ((sum(mfccs[2])/len(mfccs[2])))
    mfccs3_var = (statistics.variance(mfccs[2]))

    mfccs4_mean = ((sum(mfccs[3])/len(mfccs[3])))
    mfccs4_var = (statistics.variance(mfccs[3]))

    mfccs5_mean = ((sum(mfccs[4])/len(mfccs[4])))
    mfccs5_var = (statistics.variance(mfccs[4]))

    mfccs6_mean = ((sum(mfccs[5])/len(mfccs[5])))
    mfccs6_var = (statistics.variance(mfccs[5]))

    mfccs7_mean = ((sum(mfccs[6])/len(mfccs[6])))
    mfccs7_var = (statistics.variance(mfccs[6]))

    mfccs8_mean = ((sum(mfccs[7])/len(mfccs[7])))
    mfccs8_var = (statistics.variance(mfccs[7]))

    mfccs9_mean = ((sum(mfccs[8])/len(mfccs[8])))
    mfccs9_var = (statistics.variance(mfccs[8]))

    mfccs10_mean = ((sum(mfccs[9])/len(mfccs[9])))
    mfccs10_var = (statistics.variance(mfccs[9]))

    mfccs11_mean = ((sum(mfccs[10])/len(mfccs[10])))
    mfccs11_var = (statistics.variance(mfccs[10]))

    mfccs12_mean = ((sum(mfccs[11])/len(mfccs[11])))
    mfccs12_var = (statistics.variance(mfccs[11]))

    mfccs13_mean = ((sum(mfccs[12])/len(mfccs[12])))
    mfccs13_var = (statistics.variance(mfccs[12]))

    mfccs14_mean = ((sum(mfccs[13])/len(mfccs[13])))
    mfccs14_var = (statistics.variance(mfccs[13]))

    mfccs15_mean = ((sum(mfccs[14])/len(mfccs[14])))
    mfccs15_var = (statistics.variance(mfccs[14]))

    mfccs16_mean = ((sum(mfccs[15])/len(mfccs[15])))
    mfccs16_var = (statistics.variance(mfccs[15]))

    mfccs17_mean = ((sum(mfccs[16])/len(mfccs[16])))
    mfccs17_var = (statistics.variance(mfccs[16]))

    mfccs18_mean = ((sum(mfccs[17])/len(mfccs[17])))
    mfccs18_var = (statistics.variance(mfccs[17]))

    mfccs19_mean = ((sum(mfccs[18])/len(mfccs[18])))
    mfccs19_var = (statistics.variance(mfccs[18]))

    mfccs20_mean = ((sum(mfccs[19])/len(mfccs[19])))
    mfccs20_var = (statistics.variance(mfccs[19]))

    print('MFCC', mfccs10_mean)
    with open('/Users/BenH/Desktop/Thrive/Data/features_30_sec.csv','r') as file:
        data = file.readlines()

    print(len(data))
    data[5] = ('input_file' + ',' + '0' + ',' + str(chroma_mean) + ',' + str(chroma_var)  
    + ',' + str(rms_mean) + ',' + str(rms_var) + ',' + str(spec_cent_mean) + ',' 
    + str(spec_cent_var) + ',' + str(spec_bw_mean) + ',' + str(spec_bw_var) + ',' 
    + str(rolloff_mean) + ',' + str(rolloff_var) + ',' + str(zero_crossing_rate_mean) + ',' 
    + str(zero_crossing_rate_var) + ',' + '0' + ',' + '0' + ',' + '0' + ',' + '0' + ',' + str(tempo) + ',' + str(mfccs1_mean) 
    + ',' + str(mfccs1_var)  + ',' + str(mfccs2_mean) + ',' + str(mfccs2_var)  + ',' 
    + str(mfccs3_mean) + ',' + str(mfccs3_var)  + ',' + str(mfccs4_mean) + ',' 
    + str(mfccs4_var)  + ',' + str(mfccs5_mean) + ',' + str(mfccs5_var)  + ',' 
    + str(mfccs6_mean) + ',' + str(mfccs6_var)  + ',' + str(mfccs7_mean) + ',' 
    + str(mfccs7_var)  + ',' + str(mfccs8_mean) + ',' + str(mfccs8_var)  + ',' 
    + str(mfccs9_mean) + ',' + str(mfccs9_var)  + ',' + str(mfccs10_mean) + ',' 
    + str(mfccs10_var)  + ',' + str(mfccs11_mean) + ',' + str(mfccs11_var)  + ',' 
    + str(mfccs12_mean) + ',' + str(mfccs12_var)  + ',' + str(mfccs13_mean) + ',' 
    + str(mfccs13_var)  + ',' + str(mfccs14_mean) + ',' + str(mfccs14_var)  + ',' 
    + str(mfccs15_mean) + ',' + str(mfccs15_var)  + ',' + str(mfccs16_mean) + ',' 
    + str(mfccs16_var)  + ',' + str(mfccs17_mean) + ',' + str(mfccs17_var)  + ',' 
    + str(mfccs18_mean) + ',' + str(mfccs18_var)  + ',' + str(mfccs19_mean) + ',' 
    + str(mfccs19_var)  + ',' + str(mfccs20_mean) + ',' + str(mfccs20_var) + ',' 
    + 'input_label' + '\n')

    with open('/Users/BenH/Desktop/Thrive/Data/features_30_sec.csv','w') as file:
        file.writelines(data)
    file.close()


    

def find_similar_songs(name):

    # Read data
    data = pd.read_csv('/Users/BenH/Desktop/Thrive/Data/features_30_sec.csv', index_col='filename')


    # Extract labels
    labels = data[['label']]

    # Drop labels from original dataframe
    data = data.drop(columns=['length','label', 'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var'])
    #data.head()
    #data = data.drop(columns=['length','label'])
    data_scaled=preprocessing.scale(data)

    # Scale the data
    similarity = cosine_similarity(data_scaled)

    # Convert into a dataframe and then set the row index and column names as labels
    sim_df_labels = pd.DataFrame(similarity)
    sim_df_names = sim_df_labels.set_index(labels.index)
    sim_df_names.columns = labels.index

    sim_df_names.head()


    # Find songs most similar to another song
    series = sim_df_names[name].sort_values(ascending = False)
        
    # Remove cosine similarity == 1 (songs will always have the best match with themselves)
    series = series.drop(name)
        
    # Display the 5 top matches 
    return (series.head(1))























