import os
import glob
from IPython.display import Image, Audio, display
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
# detector de anomalias
from sklearn.ensemble import IsolationForest

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
fig, ax = plt.subplots(len(dados), 1, figsize=(15, 3*len(dados)))
for i, linha in enumerate(dados):
    ax[i].set_title(bons[i])
    for x, y in linha:
        ax[i].plot(x, y)
plt.show()


X = []
for i, linha in enumerate(dados):
    for _, y in linha:
        X.append(y)

clf = IsolationForest(contamination=0.15)
clf.fit(X)
ano = clf.predict(X)
ano = (ano == -1)

fig, ax = plt.subplots(len(dados), 1, figsize=(15, 3*len(dados)))
n = -1
for i, linha in enumerate(dados):
    ax[i].set_title(bons[i])
    for x, y in linha:
        n += 1
        ax[i].plot(x, y, color='red' if ano[n] else 'blue')
plt.show()
