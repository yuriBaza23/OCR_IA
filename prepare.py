from sklearn.model_selection import train_test_split as prepareTestAndTrain

# Usamos a função train_test_split do sklearn para separar os dados de treino e teste.
def prepare(feat, label):
  FeatTrain, FeatTest, labelTrain, labelTest = prepareTestAndTrain(feat, label, test_size=0.30, random_state=42)

  return (FeatTrain, FeatTest, labelTrain, labelTest)