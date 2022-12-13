from PIL import Image

# Para calcular intevalos entre pixels pretos e brancos
from calcInterval import calcInterval

import glob
import numpy as np
import string

# Abre os arquivos com um limite. Também recebe o diretório onde os arquivos estão
# No caso dessa aplicação, não há divisões em arquivos de teste e treino.

# OBS: Nessa função se calcula o maior e o menor intervalo entre os pixels brancos e pretos
# através de outra função criada em outro arquivo. Os valores extraídos são adicionados nas
# features. O index é o rotulo do arquivo. Isso significa que os arquivos contidos na pasta A
# do diretório informado, terão o rotulo 0. Os arquivos contidos na pasta B do diretório informado,
# terão o rotulo 1. E assim por diante.
def openFiles(limit, mainDir, dir, values):
    file = open("./results/featuresAndLabels.txt", 'w')
    file.writelines("Arquivo correspondente as features e labels obtidas:\n\n")

    fileNames = []
    path = ''

    # Rotulos e Features (y para rótulos e X para features)
    y = []
    X = []

    # Letras de A a Z em maiúsculo. Será utilizado para percorrer os diretórios
    letters = values

    if(dir != ''):
        path = './' + mainDir + '/' + dir + '/{}/*.bmp'
    else:
        path = './' + mainDir + '/{}/*.bmp'

    # Caso o limite seja maior que 0, ele abre os arquivos com o limite
    if limit > 0:
        for index, letter in enumerate(letters):
            fileNames = glob.glob(path.format(letter))

            for name in fileNames[:limit]:
                im = Image.open(name, 'r')
                px = list(im.getdata())
                
                values = []
                n = int(len(px)/9)
                i = int(len(px)/9)
                j = 0

                while i <= len(px):
                    whitePixels = np.sum(px[j:i])/len(px[j:i])
                    blackPixels = (len(px) - whitePixels)/len(px[j:i])
                    
                    greaterRangeBlack = calcInterval(px[j:i], 'maior')[0]/len(px[j:i])
                    greaterRangeWhite = calcInterval(px[j:i], 'maior')[1]/len(px[j:i])
                    
                    smallestRangeBlack, smallestRangeWhite = calcInterval(px[j:i], 'menor', blackPixels, whitePixels)
                    smallestRangeBlack = smallestRangeBlack/len(px[j:i])
                    smallestRangeWhite = smallestRangeWhite/len(px[j:i])
                    
                    averageSumGreater = (greaterRangeBlack + greaterRangeWhite)/len(px)
                    averageSumSmallest = (smallestRangeBlack + smallestRangeWhite)/len(px)
                
                    values.append(whitePixels) # Adiciona a quantidade de pixels brancos
                    values.append(blackPixels) # Adiciona a quantidade de pixels pretos
                    values.append(greaterRangeBlack) # Adiciona o maior intervalo entre pixels pretos
                    values.append(greaterRangeWhite) # Adiciona o maior intervalo entre pixels brancos
                    values.append(smallestRangeBlack) # Adiciona o menor intervalo entre pixels pretos
                    values.append(smallestRangeWhite) # Adiciona o menor intervalo entre pixels brancos
                    values.append(averageSumGreater) # Adiciona a média entre o maior intervalo entre pixels pretos e brancos
                    values.append(averageSumSmallest) # Adiciona a média entre o menor intervalo entre pixels pretos e brancos
                    
                    j = i
                    i = i + n
                    
                X.append(values)
                y.append(index)

    elif limit == 0: # Caso o limite seja igual a 0, ele abre todos os arquivos :)
        for index, letter in enumerate(letters):
            fileNames = glob.glob(path.format(letter))

            for name in fileNames:
                im = Image.open(name, 'r')
                px = list(im.getdata())
                
                values = []
                n = int(len(px)/9)
                i = int(len(px)/9)
                j = 0

                while i <= len(px):
                    whitePixels = np.sum(px[j:i])/len(px[j:i])
                    blackPixels = (len(px) - whitePixels)/len(px[j:i])
                    
                    greaterRangeBlack = calcInterval(px[j:i], 'maior')[0]/len(px[j:i])
                    greaterRangeWhite = calcInterval(px[j:i], 'maior')[1]/len(px[j:i])
                    
                    smallestRangeBlack, smallestRangeWhite = calcInterval(px[j:i], 'menor', blackPixels, whitePixels)
                    smallestRangeBlack = smallestRangeBlack/len(px[j:i])
                    smallestRangeWhite = smallestRangeWhite/len(px[j:i])
                    
                    averageSumGreater = (greaterRangeBlack + greaterRangeWhite)/len(px)
                    averageSumSmallest = (smallestRangeBlack + smallestRangeWhite)/len(px)
                
                    values.append(whitePixels) # Adiciona a quantidade de pixels brancos
                    values.append(blackPixels) # Adiciona a quantidade de pixels pretos
                    values.append(greaterRangeBlack) # Adiciona o maior intervalo entre pixels pretos
                    values.append(greaterRangeWhite) # Adiciona o maior intervalo entre pixels brancos
                    values.append(smallestRangeBlack) # Adiciona o menor intervalo entre pixels pretos
                    values.append(smallestRangeWhite) # Adiciona o menor intervalo entre pixels brancos
                    values.append(averageSumGreater) # Adiciona a média entre o maior intervalo entre pixels pretos e brancos
                    values.append(averageSumSmallest) # Adiciona a média entre o menor intervalo entre pixels pretos e brancos
                    
                    j = i
                    i = i + n
                    
                X.append(values)
                y.append(index)

    file.writelines('Features: ' + str(len(X)) + "\n")
    file.writelines('Labels: ' + str(len(y)) + "\n")
    file.close()
    return (X, y)