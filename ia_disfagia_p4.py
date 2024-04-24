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

partes_audio = []
dados_ext = []


def split_audio_and_check(input_file):
    # Carrega o áudio
    y, sr = librosa.load(input_file, sr=None)

    # Divide o áudio em partes de 0.5 segundos
    hop_length = int(0.5 * sr)
    parts = librosa.effects.split(y, hop_length=hop_length)

    for i, (start, end) in enumerate(parts):
        print(f"Parte {i+1}:")
        duration = librosa.get_duration(y=y[start:end], sr=sr)
        print(f"Duração da parte: {duration} segundos")

        # Verifica a cada 0.25 segundos
        check_step = int(0.25 * sr)
        for j in range(start, end, check_step):
            sub_audio = y[j:min(j+check_step, end)]
            amplitude = max(sub_audio) - min(sub_audio)
            time_sec = librosa.samples_to_time(j, sr=sr)
            print(f"Amplitude na posição {time_sec} segundos: {amplitude}")
            partes_audio.append(sub_audio)


def extrair_caracteristicas():
    y = partes_audio[0]
    chroma_stft = librosa.feature.chroma_stft(y=y)
    tonnetz = librosa.feature.tonnetz(y=y)
    rmse = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y)
    spec_bw = librosa.feature.spectral_bandwidth(y=y)
    rolloff = librosa.feature.spectral_rolloff(y=y)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y)
    melspectrogram = librosa.feature.melspectrogram(y=y)
    # fazer a média dos valores
    atributos = []
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

    dados_ext.append(atributos)


def test_modelo():
    # Carregar o modelo treinado
    modelo = joblib.load('RandomForestClassifier.joblib')

    # Normalizar os dados
    scaler = MinMaxScaler()
    dados_normalizados = scaler.fit_transform(dados_ext)

    # Selecionar as melhores características
    selecao_caracteristicas = SelectKBest(chi2, k=10)
    dados_selecionados = selecao_caracteristicas.fit_transform(
        dados_normalizados)

    # Fazer a previsão usando o modelo
    previsao = modelo.predict(dados_selecionados)

    # Definir o valor de y_true
    y_true = ...

    # Imprimir o relatório de classificação
    print(classification_report(previsao, y_true))


# Substitua 'caminho/para/seu/arquivo.mp3' pelo caminho do seu arquivo de áudio
split_audio_and_check("./a00659.mp3")
extrair_caracteristicas()
test_modelo()


# for parte in partes_audio:
#     print(f"parte: {parte} \n")
