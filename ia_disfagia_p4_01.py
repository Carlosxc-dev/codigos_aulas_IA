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

X_train = []  # Para as amostras de treinamento
Y_train = []  # Para as classes correspondentes

X_test = []   # Para as amostras de teste
Y_test = []   # Para as classes verdadeiras

# Treinamento do modelo
for i, linha in enumerate(dados):
    for _, y in linha:
        X_train.append(y)
        Y_train.append(-1)  # -1 para anomalia

# Predição
for i, linha in enumerate(dados):
    for _, y in linha:
        X_test.append(y)
        Y_test.append(-1)  # -1 para anomalia

clf = RandomForestClassifier()
clf.fit(X_train, Y_train)
ano = clf.predict(X_test)
ano = (ano == -1)

# Plotagem
fig, ax = plt.subplots(len(dados), 1, figsize=(15, 3*len(dados)))
n = -1
for i, linha in enumerate(dados):
    ax[i].set_title(bons[i])
    for x, y in linha:
        n += 1
        ax[i].plot(x, y, color='red' if ano[n] else 'blue')
plt.show()
