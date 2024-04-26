from sklearn.ensemble import IsolationForest
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
import numpy as np
import matplotlib.pyplot as plt

audio_path = glob.glob('test/valid/*.mp3')
print(f"caminho dos audios: {audio_path}\n\n")

y, sr = librosa.load(audio_path[4])  # sr = 22050 , y = ndarray
audio = y
x = np.arange(len(y))/sr
t = np.arange(len(audio)) / sr
tamanho_intervalo = 0.5 * sr
passo = 0.25 * sr
inicio_intervalo = 0
limites = []
lim = []
dados = []

# Enquanto o início do próximo intervalo for menor que o comprimento total do áudio
while inicio_intervalo + tamanho_intervalo < len(audio):
    # Calcular o fim do intervalo
    fim_intervalo = inicio_intervalo + tamanho_intervalo

    inicio_intervalo = int(inicio_intervalo)
    fim_intervalo = int(fim_intervalo)

    # Adicionar os limites do intervalo à lista de limites
    limites.append((x[inicio_intervalo:fim_intervalo],
                   y[inicio_intervalo:fim_intervalo]))
    lim.append((inicio_intervalo, fim_intervalo))
    # Atualizar o início para o próximo intervalo
    dados.append(limites)
    inicio_intervalo += passo


array_de_dados = np.array(limites, dtype=int)

# Plotar os limites em um gráfico
plt.figure(figsize=(10, 5))
plt.plot(audio, color='blue')

# Plotar os limites
for limite in lim:
    plt.axvline(x=limite[0], color='red', linestyle='--')
    plt.axvline(x=limite[1], color='red', linestyle='--')

plt.xlabel('Amostras')
plt.ylabel('Amplitude')

plt.show()

# ---------------------------------------------------------------- aplicar no modelo

# Transformar a lista de limites em um array numpy

X = []
for i, linha in enumerate(dados):
    for x, y in linha:
        X.append(y)

# Transformar a lista de limites em um array numpy bidimensional
X = np.array(X)

clf = IsolationForest(contamination=0.20)
clf.fit(X)
ano = clf.predict(X)
ano = (ano == -1)

fig, ax = plt.subplots(1, 1, figsize=(15, 3*len(dados)))
n = -1

for linha in dados:
    for x, y in linha:
        if ano[n]:
            plt.axvline(x=x[0], color='red', linestyle='--')
            plt.axvline(x=y[1], color='red', linestyle='--')

for i, linha in enumerate(dados):
    for x, y in linha:
        n += 1
        ax.plot(x, y, color='black' if ano[n] else 'gray')
        plt.axvline(x=x[0], ymax=y[0], ymin=y[1], color='red',
                    linestyle='--' if ano[n] else None)
plt.show()
