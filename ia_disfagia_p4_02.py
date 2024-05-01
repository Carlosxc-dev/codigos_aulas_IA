from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import joblib

# carregar um arquivo de áudio
bons = glob.glob('test/valid/*.mp3')


dados = []
for b in tqdm(bons):
    nome = os.path.basename(b)[:-4]
    y, sr = librosa.load(b)
    x = np.arange(len(y))/sr

    janela = int(0.5 * sr)
    passo = int(0.25 * sr)

    linha = []
    for i in range(0, len(y), int(passo)):
        max = i + janela
        if max > len(y):
            break
        linha.append((x[i:max], y[i:max]))
    dados.append(linha)

# Plotar os dados de áudio
# fig, ax = plt.subplots(len(dados), 1, figsize=(15, 3*len(dados)))
# for i, linha in enumerate(dados):
#     ax[i].set_title(bons[i])
#     for x, y in linha:
#         ax[i].plot(x, y, color='blue')
#         for intervalo in range(0, len(x), int(passo)):
#             ax[i].axvline(x=x[intervalo], color='red',
#                           linestyle='--', linewidth=0.5)
# plt.tight_layout()
# plt.show()

# --------------------------------------------------------------
X = []
y = []
for i, linha in enumerate(dados):
    for x, audio in linha:
        X.append(audio)
        y.append(-1)  # -1 para anomalia

# Criar e carregar o classificador RandomForest
clf = joblib.load('RandomForestClassifier.joblib')
clf.fit(X, y)

# Fazer previsões e ajustar os valores para -1 (anomalia) e 1 (não anomalia)
predictions = clf.predict(X)
predictions = np.where(predictions == -1, -1, 1)

# Plotar os dados de áudio
fig, ax = plt.subplots(len(dados), 1, figsize=(15, 3*len(dados)))
for i, linha in enumerate(dados):
    ax[i].set_title(bons[i])
    for x, y in linha:
        ax[i].plot(x, y, color='blue')  # Plotar os dados de áudio em azul
        for intervalo, anomalia in zip(range(0, len(x) - janela, passo), predictions):
            print(anomalia)
            if anomalia == -1:  # Se uma anomalia for identificada
                print(anomalia)
                inicio = int(intervalo)
                fim = int(intervalo + janela)
                ax[i].axvspan(x[inicio], x[fim], color='red',
                              alpha=0.3)  # Corrigindo a coloração
plt.tight_layout()
plt.show()
