# Rede Neural Profunda com Mecanismos para Evitar o Vanishing Gradient

Este exemplo demonstra uma rede neural profunda construída com TensorFlow e Keras, implementando estratégias para evitar o problema do **vanishing gradient**. Utilizamos técnicas como a função de ativação **ReLU**, inicialização de pesos **He** e **Batch Normalization** para estabilizar os gradientes durante o treinamento.

## Código

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD
import numpy as np
```

### Importações Principais

- `tensorflow` e `tensorflow.keras`: Bibliotecas principais para criar e treinar redes neurais.
- `numpy`: Utilizada para gerar dados de exemplo aleatórios.

---

### 1. Configurações do Exemplo

```python
# Configurações para o exemplo
input_dim = 10  # Dimensão da entrada
hidden_units = 64  # Número de neurônios nas camadas ocultas
num_layers = 10  # Número de camadas ocultas (rede profunda)
num_samples = 1000  # Número de amostras de dados
```

- **input_dim**: Dimensão da entrada (10), ou seja, o número de características de cada amostra.
- **hidden_units**: Número de neurônios em cada camada oculta (64), definindo a capacidade de aprendizado de cada camada.
- **num_layers**: Número total de camadas ocultas (10), tornando a rede profunda.
- **num_samples**: Número de amostras de dados aleatórios gerados para o treinamento (1000).

---

### 2. Gerar Dados de Exemplo

```python
# Gerar dados de exemplo
X = np.random.rand(num_samples, input_dim)  # Dados de entrada aleatórios
y = np.random.randint(0, 2, size=(num_samples, 1))  # Rótulos binários aleatórios
```

- **X**: Matriz de dados de entrada aleatórios de dimensão (1000, 10), simulando características de entrada.
- **y**: Vetor de rótulos aleatórios (0 ou 1) para cada amostra, representando uma tarefa de classificação binária.

---

### 3. Construção do Modelo

```python
# Construir a rede neural com várias camadas ocultas e função ReLU
model = Sequential()
model.add(Dense(hidden_units, input_dim=input_dim, activation='relu', kernel_initializer='he_normal'))
```

Iniciar a rede neural como um modelo sequencial:

- **Dense(hidden_units, activation='relu')**: Adiciona uma camada densa (fully connected) com 64 neurônios e função de ativação ReLU para evitar o vanishing gradient.
- **kernel_initializer='he_normal'**: Inicialização de pesos He para estabilizar os gradientes nas camadas ocultas.

---

### 4. Adição de Camadas Ocultas com ReLU e Batch Normalization

```python
# Adiciona múltiplas camadas ocultas com ativação ReLU e Batch Normalization
for _ in range(num_layers):
    model.add(Dense(hidden_units, activation='relu', kernel_initializer='he_normal'))
    model.add(BatchNormalization())  # Normalização por camada para estabilizar os gradientes
```

Este FOR adiciona múltiplas camadas densas e normalização por batch:

- **BatchNormalization()**: Normaliza as ativações em cada camada, o que ajuda a estabilizar os gradientes e facilita o treinamento em redes profundas.

---

### 5. Camada de Saída

```python
# Camada de saída
model.add(Dense(1, activation='sigmoid'))
```

- A camada de saída possui um único neurônio com função de ativação **sigmoid**, adequada para problemas de classificação binária. Ela mapeia a saída para um valor entre 0 e 1.

---

### 6. Compilação do Modelo

```python
# Compilar o modelo com gradiente descendente (SGD) para observar a melhora no treinamento
model.compile(optimizer=SGD(learning_rate=0.01), loss='binary_crossentropy')
```

- **optimizer=SGD(learning_rate=0.01)**: Utiliza o otimizador SGD (Stochastic Gradient Descent) com uma taxa de aprendizado de 0.01.
- **loss='binary_crossentropy'**: Função de perda para problemas de classificação binária, calculando o erro entre a previsão e o valor verdadeiro.

---

### 7. Treinamento do Modelo

```python
# Treinar o modelo por poucas épocas para observar o efeito do treinamento melhorado
history = model.fit(X, y, epochs=10, verbose=1)
```

- **epochs=10**: O modelo treina por 10 épocas para observar o impacto da normalização e do ReLU sobre a perda.

---

### 8. Resultados do Treinamento

```python
# Mostrar os valores de perda ao longo do treinamento
print("Perda durante o treinamento:", history.history['loss'])
```

- Imprime a perda em cada época, permitindo avaliar a convergência e eficácia das técnicas aplicadas para evitar o vanishing gradient.

---

## Observação dos Resultados

Ao implementar ReLU, Batch Normalization e inicialização He, o modelo consegue evitar o vanishing gradient, permitindo que o treinamento avance de forma mais eficaz e que a perda diminua consistentemente ao longo das épocas.

Importante destacar que se você rodar o arquivo `Raw_code.py` e depois o `Complete_code.py` irá perceber que as perdas nos resultados são menores desde o inicio no `Raw_code.py`. Porém se você reparar durante os outros epochs desse código, as perdas continuam semelhantes mudando bem pouco, ou seja mostra que o modelo não está se ajustando adequadamente para reduzir o erro.
  
  * Exemplo: [0.7137, 0.6947, 0.6937, 0.6935, 0.6935, 0.6933, 0.6932, 0.6939, 0.6938, 0.6939]


Já no `Complete_code.py` podemos ver mesmo que uma perda no resultado maior desde o início, o modelo vai reduzindo cada vez mais a cada epoch esse erro, mostrando que as funções de ReLU e a normalização estão sendo eficazes em ajustar adequadamente o modelo para reduzir o erro.

  * Exemplo: [0.8749, 0.7958, 0.7442, 0.7272, 0.7449, 0.7306, 0.7244, 0.7344, 0.7100, 0.7005]


---
