from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix as matriz

import numpy as np

# Classificador SVM. Utilizamos a função SVC do sklearn para classificar os dados e
# a função confusion_matrix para gerar a matriz de confusão.
def svm(FeatTrain, FeatTest, labelTrain, labelTest):
  file = open("./results/svm.txt", 'w')

  clf = SVC(probability=True)

  clf.fit(np.array(FeatTrain), labelTrain)

  svmProbs = clf.predict_proba(FeatTest)

  acuracia = clf.score(np.array(FeatTest), labelTest)

  labelPred = clf.predict(FeatTest)

  matrix = matriz(labelTest, labelPred)

  file.writelines("Arquivo correspondente ao SVM:\n\n")

  a = []

  for x in range(len(matrix)):
    a.append([])
    for j in matrix[x]:
      a[x].append(j)
          
  for x in a:
    file.writelines(str(x) + "\n")

  file.writelines("\n\nAcurácia =" + str(acuracia) + "%.")

  file.close()

  return svmProbs