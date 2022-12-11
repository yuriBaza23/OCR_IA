# Calcula o intervalo de pixels de maior e menor quantidade de pixels pretos.
# Para isso é necessário especificar se queremos o maior ou menor intervalo.
def calcInterval(pixels, type, one=0, two=0):
  atual = pixels[0]
  qtd0 = one
  qtd1 = two
  qtd = 0
  if type == 'maior':
    for i, values in enumerate(pixels):
      if values == atual:
        qtd += 1
      else:
        if atual == 0:
          if qtd0 < qtd:
            qtd0 = qtd
        else:
          if qtd1 < qtd:
            qtd1 = qtd
        atual = values
        qtd = 1
      if i == len(pixels)-1:
        if atual == 1:
          if qtd1 < qtd:
            qtd1 = qtd
        else:
          if qtd0 < qtd:
            qtd0 = qtd
    return (qtd0, qtd1)
  elif type == 'menor':
    for i, values in enumerate(pixels):
      if values == atual:
        qtd += 1
      else:
        if atual == 0:
          if qtd0 > qtd:
            qtd0 = qtd
        else:
          if qtd1 > qtd:
            qtd1 = qtd
        atual = values
        qtd = 1
      if i == len(pixels)-1:
        if atual == 1:
          if qtd1 > qtd:
            qtd1 = qtd
        else:
          if qtd0 > qtd:
            qtd0 = qtd
    return (qtd0, qtd1)
