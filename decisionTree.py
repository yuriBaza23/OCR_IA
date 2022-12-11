from sklearn.tree import DecisionTreeClassifier as tree
from sklearn.metrics import confusion_matrix as matriz

import numpy as np

# Classificador Decision Tree.
# Recebe as features de treino, as features de teste, os rótulos de treino e os rótulos de teste.
# Usamos o DecisionTreeClassifier do sklearn para criar o classificador.
def DT(FeatTrain, FeatTest, labelTrain, labelTest):
  file = open("./results/decisionTree.txt", 'w')
  clf = tree()

  clf.fit(np.array(FeatTrain), labelTrain)

  dtProbs = clf.predict_proba(FeatTest)

  acuracia = clf.score(np.array(FeatTest), labelTest)

  labelPred = clf.predict(FeatTest)

  matrix = matriz(labelTest, labelPred)

  file.writelines("Matriz de confusão para Decision Tree:\n\n")

  a = []

  for x in range(len(matrix)):
      a.append([])
      for j in matrix[x]:
          a[x].append(j)
          
  for x in a:
    file.writelines(str(x) + "\n")

  file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")

  file.close()

  return dtProbs