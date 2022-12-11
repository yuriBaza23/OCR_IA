# Materia: Inteligência Artificial
# Professor: Dr. Diego Bertolini
# Alunos: Yuri Baza e Vinicius Simões

import openFiles as of
import prepare as pr
import normalization as nr

from knn import knn
from svm import svm
from decisionTree import DT as dt
from mlp import MLP as mlp
from randomForest import RF as rf
from sr import sumRule as sr

def main():
    # Lista com as probabilidades de cada classificador.
    probs = []
    FeatTrain, FeatTest, labelTrain, labelTest = [], [], [], []
    Features, labels = [], []
    
    # PARA LETRAS -----------
    usingPrepare = True
    execSVM = False
    values = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    mainDir = 'images'

    # PARA NUMEROS -----------
    # usingPrepare = False
    # execSVM = True
    # mainDir = 'images2'
    # values = '0123456789'
    # trainDir = 'train'
    # testDir = 'test'


    # Abrimos os arquivos com os dados de treino e teste.
    # Como resposta obtemos as features e as labels.
    print("Abrindo arquivos (pasta images)...\n")
    if usingPrepare == True:
        Features, labels = of.openFiles(0, mainDir, '', values)
    else:
        FeatTrain, labelTrain = of.openFiles(0, mainDir, trainDir, values)
        FeatTest, labelTest = of.openFiles(0, mainDir, testDir, values)

    # Preparamos os arquivos de teste e treino conforme os
    # dados obtidos no passo anterior.
    print("Preparando testes e treinos (arquivo prepare)...\n")
    if usingPrepare == True: FeatTrain, FeatTest, labelTrain, labelTest = pr.prepare(Features, labels)

    # Normalizamos os dados de treino e teste.
    # Aqui escolhemos o zscore como método de normalização.
    print("Normalizando os dados...")
    FeatTrain, FeatTest = nr.zScore(FeatTrain, FeatTest)
    
    # Começamos a aplicar os classificadores.
    # O primeiro que usamos foi o KNN
    # Passamos as features e labels de treino e teste.
    # Aqui escolhemos o k = 1, 3, 5, 7, 9 e 11.
    print("Aplicando KNN...")
    knnProbs = []
    knnParameters = [1, 3, 5, 7, 9, 11]

    # Para facilitar o código, criamos uma lista com os resultados e
    # outra lista com os parâmetros do KNN.
    # Ao final, somamos todos os resultados para obter a probabilidade.
    for i in range(6):
        knnProbs.append(knn(FeatTrain, FeatTest, labelTrain, labelTest, knnParameters[i])) 

    probs.append(knnProbs[0] + knnProbs[1] + knnProbs[2] + knnProbs[3] + knnProbs[4] + knnProbs[5])
    print('\n')

    # Aplicamos o Decision Tree.
    # Passamos as features e labels de treino e teste.
    print("Aplicando Decision Tree...")
    probs.append(dt(FeatTrain, FeatTest, labelTrain, labelTest))

    # Aplicamos o Multi Layer Perceptron.
    # Passamos as features e labels de treino e teste.
    print("Aplicando Multi Layer Perceptron...")
    probs.append(mlp(FeatTrain, FeatTest, labelTrain, labelTest))

    # Aplicamos o Random Forest.
    # Passamos as features e labels de treino e teste.
    print("Aplicando Random Forest...")
    probs.append(rf(FeatTrain, FeatTest, labelTrain, labelTest))
    
    if execSVM == True:
        # Aplicamos o SVM.
        # Passamos as features e labels de treino e teste.
        print("Aplicando SVM...")
        probs.append(svm(FeatTrain, FeatTest, labelTrain, labelTest))
    else:
        probs.append(0)

    # Somamos todas as probabilidades para obter a probabilidade final.
    allProbs = probs[0] + probs[1] + probs[2] + probs[3] + probs[4]

    # Aplicamos a regra da soma.
    # Passamos as labels de teste e as probabilidades finais.
    print("Aplicando regra da soma...")
    sr(labelTest, allProbs)

    print("Fim do programa.")
    
if __name__ == "__main__":
    main()