# Identificar os períodos de deglutição nos arquivos válidos

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from tqdm import tqdm
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from IPython.display import Image, Audio, display
from ipywidgets import widgets
import joblib
from sklearn.tree import export_text
import warnings
from scipy.io import wavfile

# lendo planilha de parâmetros dos audios
df = pd.read_csv('arquivos.csv')
df.set_index('0',inplace=True)
df.head()

# lendo planilha de classificação de anomalias
tutor = pd.read_csv('classificacao_anomalias.csv')
tutor = tutor[['File Name','Label']]
tutor['Label'] = tutor['Label'].apply(lambda x: 1 if x == 0 else 0)
tutor.set_index('File Name',inplace=True)

# identificando labels e quantidades
display(tutor['Label'].value_counts())
print(tutor.head())

# conectando a tabela de parâmetros da etapa 1 com os sons válidos do tutor
ds = df.join(tutor)

# removo as linhas que não tem identificação de válido
ds.dropna(inplace=True, axis=0)
display(ds.shape)
print((ds.head()))

# Utilizando metodologias de aprendizado supervisionado classificador 
#com validação cruzada (CV) para avaliação das técnicas com metrica F1:

# removo a coluna de nome do arquivo
X = ds.drop(columns=['Label']).values
y = ds['Label'].values
X.shape, y.shape


Xp = MinMaxScaler().fit_transform(X)
skb = SelectKBest(chi2, k=50)
X = skb.fit_transform(Xp, y)
del Xp
X.shape, skb.get_params()


warnings.filterwarnings('ignore')

max_depth = 5

classificadores = [
    DecisionTreeClassifier(random_state=42,max_depth=max_depth),
    ExtraTreeClassifier(random_state=42,max_depth=max_depth),
    RandomForestClassifier(random_state=42,max_depth=max_depth),
    ExtraTreesClassifier(random_state=42,max_depth=max_depth),
    GradientBoostingClassifier(random_state=42,max_depth=max_depth),
    AdaBoostClassifier(random_state=42),
    HistGradientBoostingClassifier(random_state=42,max_depth=max_depth),
    LogisticRegression(random_state=42),
]

resultados = []
for cls in tqdm(classificadores):
    res = cross_validate(cls, X, y, cv=5, scoring='f1')
    resultados.append(
        {'metodo': cls.__class__.__name__, 
         'f1': res['test_score'].mean(), 
         'tempo': res['fit_time'].mean(),
         }
    )

warnings.filterwarnings('default')
df_res = pd.DataFrame(resultados)
x = df_res.sort_values('f1', ascending=False)
print(x)

clf = RandomForestClassifier(random_state=42,max_depth=max_depth)
clf.fit(X, y)

joblib.dump(clf, 'RandomForestClassifier.joblib')

print(clf.score(X,y))
clf = DecisionTreeClassifier(random_state=42,max_depth=max_depth)
clf.fit(X, y)

joblib.dump(clf, 'DecisionTreeClassifier.joblib')

print('Acurácia:',clf.score(X,y))

print(export_text(clf))

ds.reset_index(inplace=True)

blocos = []
for i, row in ds.head(50).iterrows():
    nome = row['0'].split('.')[0]
    out = widgets.Output()
    with out:
        display(Image(f'dataset/Charts/{nome}.png'))
        display('Inválido' if row['Label'] == 0 else 'Válido')
        display(Audio(f'G:/Meu Drive/FACULDADE/7 periodo/IA/ConsultasDataset/MP3/{nome}.mp3'))
    blocos.append(out)
print(widgets.HBox(blocos))