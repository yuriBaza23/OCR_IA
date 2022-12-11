from sklearn.neural_network import MLPClassifier as MPC
from sklearn.metrics import confusion_matrix as matriz

import numpy as np

# Classificador MPL.
# Recebe as features de treino, as features de teste, os rótulos de treino e os rótulos de teste.
# Usamos o classificador MLPClassifier do sklearn.
def MLP(FeatTrain, FeatTest, labelTrain, labelTest):
    file = open("./results/mlp.txt", 'w')
    clf = MPC(random_state=0, max_iter=10000)

    clf.fit(np.array(FeatTrain), labelTrain)

    mplProbs = clf.predict_proba(FeatTest)

    acuracia = clf.score(np.array(FeatTest), labelTest)

    labelPred = clf.predict(FeatTest)

    matrix = matriz(labelTest, labelPred)

    file.writelines("Matriz de confusão para MLP:\n\n")

    a = []

    for x in range(len(matrix)):
        a.append([])
        for j in matrix[x]:
            a[x].append(j)
            
    for x in a:
        file.writelines(str(x) + "\n")
        
    file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")

    file.close()

    return mplProbs