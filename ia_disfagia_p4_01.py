from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from xgboost import XGBClassifier
from time import time
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
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
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

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
tutor = [
    {"nome": "a00028", "d": [(1.5, 2.2), (4.7, 5.4), (7.1, 7.8)]},
    {"nome": "a00073", "d": [(4.5, 5.1)]},
    {"nome": "a00083", "d": [(3.1, 3.7)]},
    {"nome": "a00111", "d": [(1.1, 1.7), (3.5, 4.1)]},
    {"nome": "a00001", "d": [(1.5, 2.7), (5.8, 6)]},
    {"nome": "a00003", "d": [(1.5, 2.6)]},
    {"nome": "a00007", "d": [(1.6, 2.2), (4.4, 5.4)]},
]

dados = []
for t in tqdm(tutor):
    nome = t['nome']
    nome = f'test/valid/{nome}.mp3'
    y, sr = librosa.load(nome)
    x = np.arange(len(y))/sr

    janela = int(0.2 * sr)
    passo = int(0.1 * sr)

    linha = []
    for i in range(0, len(y), int(passo)):
        max = i + janela
        if max > len(y):
            break
        xp = x[i:max]
        yp = y[i:max]
        k = 0
        for ini, fim in t['d']:
            if xp[0] >= ini and xp[-1] <= fim:
                k = 1
                break

        linha.append((xp, yp, k))

    dados.append(linha)

# fig, ax = plt.subplots(len(dados), 1, figsize=(15, 6*len(dados)))
# n = -1
# for i, linha in enumerate(dados):
#     ax[i].set_title(tutor[i]['nome'])
#     for x, y, k in linha:
#         n += 1
#         ax[i].plot(x, y, color='red' if k == 1 else 'blue')
# plt.show()

# --------------------------------------------------------------

X = []
y = []
for linha in dados:
    for a, b, c in linha:
        X.append(b)
        y.append(c)


# pca = PCA(random_state=42)
# Xr = pca.fit_transform(X)

# classificadores = [
#     RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5),
#     DecisionTreeClassifier(max_depth=5, random_state=42),
#     ExtraTreeClassifier(max_depth=5, random_state=42),
#     LogisticRegression(random_state=42),
#     SVC(random_state=42),
#     LinearSVC(random_state=42),
#     NuSVC(nu=0.1, random_state=42),
#     GaussianNB(),
#     xgb.XGBClassifier(),
# ]

# resp = []
# for clf in classificadores:
#     t0 = time()
#     scores = cross_val_score(clf, Xr, y, cv=5, scoring='f1', n_jobs=-1)
#     t1 = time()
#     resp.append({'tecnica': clf.__class__.__name__,
#                 'f1': sum(scores)/len(scores), 'tempo': t1-t0})


# df = pd.DataFrame(resp)
# df.sort_values('f1', ascending=False)
# print(df)

# --------------------------------------------------------------
# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Create a pipeline with PCA and the chosen classifier
pipe = Pipeline([('scaler', StandardScaler()),          # Step 1 - normalize data
                 # Step 2 - reduce dimensionality
                 ('pca', PCA()),
                #  ('clf', RandomForestClassifier()),     # Step 3 - classifier
                 ('xgb', XGBClassifier())              # Step 3 - classifier
                 ])

# Treinando a pipeline
pipe.fit(X_train, y_train)

# Fazendo previsões nos dados de treinamento
y_train_pred = pipe.predict(X_train)

# Calculando a métrica F1 nos dados de treinamento
f1_train = f1_score(y_train, y_train_pred, average='weighted')
print("F1 Score (treinamento):", f1_train)

# Fazendo previsões nos dados de teste
y_test_pred = pipe.predict(X_test)

# Calculando a métrica F1 nos dados de teste
f1_test = f1_score(y_test, y_test_pred, average='weighted')
print("F1 Score (teste):", f1_test)


# Imprimindo os resultados
print("F1 Score (treinamento):", f1_train)
print("F1 Score (teste):", f1_test)

# Criando um gráfico
labels = ['Treinamento', 'Teste']
f1_scores = [f1_train, f1_test]

plt.bar(labels, f1_scores, color=['blue', 'green'])
plt.title('F1 Score')
plt.xlabel('Conjunto de dados')
plt.ylabel('F1 Score')
plt.show()
