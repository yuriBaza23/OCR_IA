from sklearn.neighbors import KNeighborsClassifier as Knn
from sklearn.metrics import confusion_matrix as matriz

import numpy as np

# Classificador KNN.
# Utilizamos a função KNeighborsClassifier do sklearn para classificar os dados e
# a função confusion_matrix para gerar a matriz de confusão.
def knn(FeatTrain, FeatTest, labelTrain, labelTest, i):
  file = open("./results/knn" + str(i) + ".txt", 'w')
  
  clf = Knn(i)

  clf.fit(np.array(FeatTrain), labelTrain)

  knnProbs = clf.predict_proba(FeatTest)

  acuracia = clf.score(np.array(FeatTest), labelTest)

  labelTred = clf.predict(FeatTest)

  matrix = matriz(labelTest, labelTred)

  print("Escrevendo arquivo para o KNN com k =", i)
  file.writelines("Matriz de confusão para knn k = " + str(i) + "\n\n")

  a = []

  for x in range(len(matrix)):
    a.append([])
    for j in matrix[x]:
      a[x].append(j)
          
  for x in a:
    file.writelines(str(x) + "\n")

  file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")

  file.close()

  return knnProbs