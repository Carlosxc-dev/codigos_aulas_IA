import os
import pandas as pd
import librosa
import numpy as np
import matplotlib.pyplot as plt
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

# plotar os dados
# fig, ax = plt.subplots(len(dados), 1, figsize=(15, 3*len(dados)))
# for i, linha in enumerate(dados):
#     ax[i].set_title(bons[i])
#     for x, y in linha:
#         ax[i].plot(x, y)
# plt.show()


# aplicando o modelo de detecção de anomalias
X = []
for i, linha in enumerate(dados):
    for x, y in linha:
        X.append(y)

# Transformar a lista de limites em um array numpy bidimensional
X = np.array(X)

Xp = MinMaxScaler().fit_transform(X)
skb = SelectKBest(chi2, k=50)
X = skb.fit_transform(Xp, y)
del Xp
X.shape, skb.get_params()


clf = joblib.load('RandomForestClassifier.joblib')
ano = clf.predict(X)
ano = (ano == -1)

fig, ax = plt.subplots(1, 1, figsize=(15, 3*len(dados)))
n = -1
for i, linha in enumerate(dados):
    for x, y in linha:
        n += 1
        ax.plot(x, y, color='black' if ano[n] else 'gray')
plt.show()
