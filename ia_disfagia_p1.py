# Identifica na base de dados arquivos válidos e inválidos para treinamento e testes;

# lendo mp3 e extraindo parâmetros de áudio
import glob
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

mp3s = glob.glob('G:/Meu Drive/FACULDADE/7 periodo/IA/ConsultasDataset/MP3/*.mp3')

dados = []
for mp3 in tqdm(mp3s):
    nome = os.path.basename(mp3)
    y, sr = librosa.load(mp3)
    # extrair caracteristicas de áudio
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
    atributos.extend(np.mean(chroma_stft,axis=1))
    atributos.extend(np.mean(tonnetz,axis=1))
    atributos.extend(np.mean(rmse,axis=1))
    atributos.extend(np.mean(spec_cent,axis=1))
    atributos.extend(np.mean(spec_bw,axis=1))
    atributos.extend(np.mean(rolloff,axis=1))
    atributos.extend(np.mean(zcr,axis=1))
    atributos.extend(np.mean(mfcc,axis=1))
    atributos.extend(np.mean(melspectrogram,axis=1))

    # fazer o desvio padrão dos valores
    atributos.extend(np.std(chroma_stft,axis=1))
    atributos.extend(np.std(tonnetz,axis=1))
    atributos.extend(np.std(rmse,axis=1))
    atributos.extend(np.std(spec_cent,axis=1))
    atributos.extend(np.std(spec_bw,axis=1))
    atributos.extend(np.std(rolloff,axis=1))
    atributos.extend(np.std(zcr,axis=1))
    atributos.extend(np.std(mfcc,axis=1))
    atributos.extend(np.std(melspectrogram,axis=1))

    # fazer a variação dos valores
    atributos.extend(np.ptp(chroma_stft,axis=1))
    atributos.extend(np.ptp(tonnetz,axis=1))
    atributos.extend(np.ptp(rmse,axis=1))
    atributos.extend(np.ptp(spec_cent,axis=1))
    atributos.extend(np.ptp(spec_bw,axis=1))
    atributos.extend(np.ptp(rolloff,axis=1))
    atributos.extend(np.ptp(zcr,axis=1))
    atributos.extend(np.ptp(mfcc,axis=1))
    atributos.extend(np.ptp(melspectrogram,axis=1))

    # adicionar no lista de dados
    dados.append(atributos)

# criar um dataframe
df = pd.DataFrame(dados)
x = df.head()
print(x)
df.to_csv('arquivos.csv',index=False)
