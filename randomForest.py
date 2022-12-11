from sklearn.ensemble import RandomForestClassifier as forest
from sklearn.metrics import confusion_matrix as matriz

import numpy as np

# Classificador Random Forest
def RF(FeatTrain, FeatTest, labelTrain, labelTest):
    file = open("./results/randomForest.txt", 'w')
    clf = forest(n_estimators = 100)

    clf.fit(np.array(FeatTrain), labelTrain)

    rfProbs = clf.predict_proba(FeatTest)

    acuracia = clf.score(np.array(FeatTest), labelTest)

    labelPred = clf.predict(FeatTest)

    matrix = matriz(labelTest, labelPred)

    file.writelines("Matriz de confusão para Random Forest:\n\n")

    a = []

    for x in range(len(matrix)):
        a.append([])
        for j in matrix[x]:
            a[x].append(j)
            
    for x in a:
        file.writelines(str(x) + "\n")

    file.writelines("\n\nAcurácia de: " + str(acuracia) + "%.")

    file.close()

    return rfProbs