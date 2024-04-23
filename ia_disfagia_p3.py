from glob import glob
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import joblib
from sklearn.metrics import classification_report
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import os


testes = glob('test/*/*.mp3')

validar = [[a, a.split('\\')[1]] for a in testes]
print(validar)

validar = [(a, 1 if a.split('\\')[1] == 'valid' else 0) for a in testes]
print(validar)

# lendo mp3 e extraindo parâmetros de áudio
dados = []
for mp3, alvo in tqdm(validar):
    nome = os.path.basename(mp3)
    y, sr = librosa.load(mp3)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    melspectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    # fazer a média dos valores
    atributos = [nome]
    atributos.extend(np.mean(chroma_stft, axis=1))
    atributos.extend(np.mean(tonnetz, axis=1))
    atributos.extend(np.mean(rmse, axis=1))
    atributos.extend(np.mean(spec_cent, axis=1))
    atributos.extend(np.mean(spec_bw, axis=1))
    atributos.extend(np.mean(rolloff, axis=1))
    atributos.extend(np.mean(zcr, axis=1))
    atributos.extend(np.mean(mfcc, axis=1))
    atributos.extend(np.mean(melspectrogram, axis=1))

    atributos.extend(np.std(chroma_stft, axis=1))
    atributos.extend(np.std(tonnetz, axis=1))
    atributos.extend(np.std(rmse, axis=1))
    atributos.extend(np.std(spec_cent, axis=1))
    atributos.extend(np.std(spec_bw, axis=1))
    atributos.extend(np.std(rolloff, axis=1))
    atributos.extend(np.std(zcr, axis=1))
    atributos.extend(np.std(mfcc, axis=1))
    atributos.extend(np.std(melspectrogram, axis=1))

    atributos.extend(np.ptp(chroma_stft, axis=1))
    atributos.extend(np.ptp(tonnetz, axis=1))
    atributos.extend(np.ptp(rmse, axis=1))
    atributos.extend(np.ptp(spec_cent, axis=1))
    atributos.extend(np.ptp(spec_bw, axis=1))
    atributos.extend(np.ptp(rolloff, axis=1))
    atributos.extend(np.ptp(zcr, axis=1))
    atributos.extend(np.ptp(mfcc, axis=1))
    atributos.extend(np.ptp(melspectrogram, axis=1))
    atributos.append(alvo)

    dados.append(atributos)

df = pd.DataFrame(dados)
print(df.shape)
print(df.head())

clf = joblib.load('RandomForestClassifier.joblib')

df.set_index(0, inplace=True)
X = df.drop(514, axis=1).values
y = df[514].values


Xp = MinMaxScaler().fit_transform(X)
skb = SelectKBest(chi2, k=50)
X = skb.fit_transform(Xp, y)
del Xp
X.shape, skb.get_params()

y_pred = clf.predict(X)

print(classification_report(y, y_pred))
