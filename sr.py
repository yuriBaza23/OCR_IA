from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as matriz

import numpy as np
def sumRule(labelTest, allProbs):
    file = open("./results/sumRule.txt", 'w')

    labelPred = np.argmax(allProbs, axis=1)

    acuracia = accuracy_score(labelTest, labelPred)

    matrix = matriz(labelTest, labelPred)

    file.writelines("Arquivo correspondente a regra da soma:\n\n")

    a = []

    for x in range(len(matrix)):
        a.append([])
        for j in matrix[x]:
            a[x].append(j)
        
    for x in a:
        file.writelines(str(x) + "\n")

    file.writelines("\n\nAcurácia =" + str(acuracia) + "%.")

    file.close()