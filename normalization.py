from sklearn.preprocessing import StandardScaler as zscore

# Normalizamos os dados de treino e teste.
# Usamos o zscore do sklearn para normalizar os dados.
def zScore(FeatTrain, FeatTest):
  normalize = zscore()

  FeatTrainNormalize = []

  for i, x in enumerate(normalize.fit_transform(FeatTrain)):
    FeatTrainNormalize.append([])
    for j in x:
      FeatTrainNormalize[i].append(j)

  FeatTestNormalize = []

  for i, x in enumerate(normalize.fit_transform(FeatTest)):
    FeatTestNormalize.append([])
    for j in x:
      FeatTestNormalize[i].append(j)

  return (FeatTrainNormalize, FeatTestNormalize)