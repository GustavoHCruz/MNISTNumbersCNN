# Dependências
# !pip install pandas
# !pip install tensorflow
# !pip install matplotlib
# !pip install pillow

# Importando as bibliotecas

import tensorflow as tf
import keras as keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd

from keras.layers import Conv2D, MaxPool2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizer_v2.adam import Adam
from keras.metrics import Accuracy, Precision, Recall
from keras.utils import np_utils
from PIL import Image, ImageChops

# Lendo o dataset

train_df = pd.read_csv('datasets/mnist_train.csv')
test_df = pd.read_csv('datasets/mnist_test.csv')

# Separando os dados em samples e labels

train = {'samples': [], 'labels': []}

for index, row in train_df.iterrows():
  train['samples'].append((row[1:] / 255).to_list())
  train['labels'].append(row[0])

train['samples'] = np.array(train['samples'][0:5000]).reshape(5000, 784)
train['labels'] = np.array(train['labels'][0:5000]).reshape(5000, 1)

test = {'samples': [], 'labels': []}

for index, row in test_df.iterrows():
  test['samples'].append((row[1:] / 255).to_list())
  test['labels'].append(row[0])

test['samples'] = np.array(test['samples'][0:500]).reshape(500, 784)
test['labels'] = np.array(test['labels'][0:500]).reshape(500, 1)

# Criando a estrutura da rede neural

model = Sequential()
model.add(Dense(784))
model.add(Dense(64))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))

model.build((5000, 784))

model.summary()

# Compilação do modelo & escolhada do otimizador

opt = Adam(learning_rate=0.01)

model.compile(optimizer=opt, loss=keras.losses.CategoricalCrossentropy(), metrics=[Accuracy(), Precision(), Recall()])

# Treinando o modelo

history = model.fit(train['samples'], keras.utils.np_utils.to_categorical(train['labels']), epochs=200)

# Impressão da precisão

plt.plot(history.history['accuracy'], label='accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Aplicações do modelo em predições

index = rd.randint(0, len(test['labels']))

imagem = (test['samples'][index]).reshape(1, 28, 28)
label = test['labels'][index][0]

predicao = np.argmax(history.model.predict(imagem.reshape(1, 784)))

print("Número:", label)

print("O resultado da predição foi:", predicao)

plt.imshow(imagem[0])
plt.show()

# Aplicando em exemplos reais

zero_legivel = Image.open('exemplos/zero_legivel.jpeg')
um_legivel = Image.open('exemplos/um_legivel.jpeg')
dois_legivel = Image.open('exemplos/dois_legivel.jpeg')
dois_ilegivel = Image.open('exemplos/dois_ilegivel.jpeg')

zero_legivel_convertido = zero_legivel.convert('L')
um_legivel_convertido = um_legivel.convert('L')
dois_legivel_convertido = dois_legivel.convert('L')
dois_ilegivel_convertido = dois_ilegivel.convert('L')

zero_legivel_processado = np.array(ImageChops.invert(zero_legivel_convertido).resize((28, 28))) /255
um_legivel_processado = np.array(ImageChops.invert(um_legivel_convertido).resize((28, 28))) /255
dois_legivel_processado = np.array(ImageChops.invert(dois_legivel_convertido).resize((28, 28))) /255
dois_ilegivel_processado = np.array(ImageChops.invert(dois_ilegivel_convertido).resize((28, 28))) /255

# =========================

print("Número: 0")

predicao = np.argmax(history.model.predict(zero_legivel_processado.reshape(1, 784)))

print("O resultado da predição foi:", predicao)

plt.imshow(zero_legivel)

plt.show()

# =========================

print("Número: 1")

predicao = np.argmax(history.model.predict(um_legivel_processado.reshape(1, 784)))

print("O resultado da predição foi:", predicao)

plt.imshow(um_legivel)

plt.show()

# =========================

print("Número: 2")

predicao = np.argmax(history.model.predict(dois_legivel_processado.reshape(1, 784)))

print("O resultado da predição foi:", predicao)

plt.imshow(dois_legivel)

plt.show()

# =========================

print("Número: 2?")

predicao = np.argmax(history.model.predict(dois_ilegivel_processado.reshape(1, 784)))

print("O resultado da predição foi:", predicao)

plt.imshow(dois_ilegivel)

plt.show()
