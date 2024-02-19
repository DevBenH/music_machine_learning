import librosa
import pandas as pd
import numpy as np 
import matplotlib.pylab as plt
import librosa.display

filename = librosa.example('nutcracker')

y, sr = librosa.load(filename)

y_trimmed, _ = librosa.effects.trim(y)
pd.Series(y_trimmed).plot(figsize = (10,5), lw=1)
pd.Series(y_trimmed[10000:20000]).plot(figsize = (10,5), lw=1)
plt.show()

d = librosa.stft(y)
s_db = librosa.amplitude_to_db(np.abs(d), ref=np.max)
print(s_db.shape)

#fig, ax = plt.subplots(figsize=(10,5))
#img = librosa.display.specshow(s_db, x_axis='time', y_axis='log',ax=ax)
#ax.set_title('Spectogram Example', fontsize = 20)
#plt.show()

