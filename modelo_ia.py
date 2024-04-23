import os
import librosa
import numpy as np
import shutil

from joblib import dump, load
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

audio_files = "mp3"
# labels
degluticao = 1
nao_degluticao = 0

# Função para extrair características MFCC de um segmento de áudio
def extrair_caracteristicas(segmento_audio, n_mfcc=13):
    # Calcula os MFCCs
    mfccs = librosa.feature.mfcc(y=segmento_audio, sr=44100, n_mfcc=n_mfcc)

    # Calcula a média dos MFCCs ao longo do tempo
    mfccs_mean = np.mean(mfccs.T, axis=0)

    return mfccs_mean

# Função para detectar períodos de deglutição em um segmento de áudio
def indentificar_degluticao(segmento_audio, modelo, limiar_degluticao=0.5):
    # Extrair características MFCC do segmento de áudio
    caracteristicas_segmento = extrair_caracteristicas(segmento_audio)

    # Fazer previsões com o modelo treinado
    previsao = modelo.predict([caracteristicas_segmento])

    # Verificar se é um período de deglutição
    if previsao == 1:  # 1 representa "deglutição"
        return True
    else:
        return False


# Diretório contendo os arquivos de áudio para extração de características
path_bom = 'G:/Meu Drive/FACULDADE/7 periodo/IA/aulas/ConsultasDataset/MP3/bom'

# Avaliar a precisão do modelo
X = []
y = []

# Iterar sobre os arquivos de áudio na pasta "Util"
for arquivo in os.listdir(path_bom):
    if arquivo.endswith(".mp3"):
        caminho_arquivo = os.path.join(path_bom, arquivo)
        audio_data, sr = librosa.load(caminho_arquivo, sr=None)
        audio_data = audio_data.astype(np.float32)  # Converter para float32
        mfccs_mean = extrair_caracteristicas(audio_data)

        X.append(mfccs_mean)
        y.append(degluticao)

# Diretório contendo os arquivos de áudio "não deglutição" para extração de características
path_ruim = 'G:/Meu Drive/FACULDADE/7 periodo/IA/aulas/ConsultasDataset/MP3/ruim'

# Iterar sobre os arquivos de áudio "não deglutição" na pasta "Inutil"
for arquivo in os.listdir(path_ruim):
    if arquivo.endswith(".mp3"):
        caminho_arquivo = os.path.join(path_ruim, arquivo)
        audio_data, sr = librosa.load(caminho_arquivo, sr=None)
        audio_data = audio_data.astype(np.float32)  # Converter para float32
        mfccs_mean = extrair_caracteristicas(audio_data)

        # Adicionar as características e a classe "não deglutição" aos conjuntos X e y
        X.append(mfccs_mean)
        y.append(nao_degluticao)

# Converter para arrays numpy
X = np.array(X)
y = np.array(y)

# Dividir o conjunto de dados em conjuntos de treinamento e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Inicializar e treinar o modelo de árvore de decisão
modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
modelo.fit(X_treino, y_treino)

# Fazer previsões no conjunto de teste
previsoes = modelo.predict(X_teste)

# Avaliar a precisão do modelo
precisao = accuracy_score(y_teste, previsoes)
print("Precisão:", precisao)

# Agora você pode salvar o modelo treinado se desejar
modelo_salvo = "modelo_arvore_decisao.joblib"
dump(modelo, modelo_salvo)

# Lista para armazenar os períodos de deglutição de todos os segmentos de áudio
periodos_degluticao_total = []

# Iterar sobre os arquivos de áudio na pasta "Util"
for arquivo in os.listdir(path_bom):
    if arquivo.endswith(".mp3"):
        caminho_arquivo = os.path.join(path_bom, arquivo)

        # Segmentar o áudio em segmentos de 1 segundo
        audio_data, sr = librosa.load(caminho_arquivo, sr=None)
        audio_data = audio_data.astype(np.float32)  # Converter para float32
        duracao_segmento = 1  # em segundos
        tamanho_segmento = int(sr * duracao_segmento)
        total_segmentos = len(audio_data) // tamanho_segmento

        # Para cada segmento de áudio
        for i in range(total_segmentos):
            inicio = i * tamanho_segmento
            fim = (i + 1) * tamanho_segmento
            segmento_audio = audio_data[inicio:fim]

            # Chamar a função de detecção de deglutição
            if indentificar_degluticao(segmento_audio, modelo):
                # Se for detectado um período de deglutição, adicionar à lista total
                periodos_degluticao_total.append(segmento_audio)

# Agora 'periodos_degluticao_total' contém os segmentos de áudio identificados como períodos de deglutição
print("deglutiçoes encontradas:",
      len(periodos_degluticao_total))
