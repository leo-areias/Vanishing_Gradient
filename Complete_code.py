import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD
import numpy as np

# Configurações para o exemplo
input_dim = 10  # Dimensão da entrada
hidden_units = 64  # Número de neurônios nas camadas ocultas
num_layers = 10  # Número de camadas ocultas (rede profunda)
num_samples = 1000  # Número de amostras de dados

# Gerar dados de exemplo
X = np.random.rand(num_samples, input_dim)  # Dados de entrada aleatórios
y = np.random.randint(0, 2, size=(num_samples, 1))  # Rótulos binários aleatórios

# Construir a rede neural com várias camadas ocultas e função ReLU
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'))

# Adiciona múltiplas camadas ocultas com ativação ReLU e Batch Normalization
for _ in range(num_layers):
    model.add(Dense(hidden_units, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())  # Normalização por camada para estabilizar os gradientes

# Camada de saída
model.add(Dense(1, activation='sigmoid'))

# Compilar o modelo com gradiente descendente (SGD) para observar a melhora no treinamento
model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy')

# Treinar o modelo por poucas épocas para observar o efeito do treinamento melhorado
history = model.fit(X, y, epochs=10, verbose=1)

# Mostrar os valores de perda ao longo do treinamento
print("Perda durante o treinamento:", history.history['loss'])
