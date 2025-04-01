# Importa as bibliotecas necessárias
import torch  # Biblioteca para computação com tensores e aprendizado profundo
import torch.nn as nn  # Submódulo para criar redes neurais
import torch.optim as optim  # Submódulo para otimizadores

# Define os dados de entrada (x) e os valores esperados de saída (y)
x = torch.tensor([[5.0],[10.0],[10.0],[5.0], [10.0],
                [5.0], [10.0], [10.0], [5.0], [10.0],
                [5.0], [10.0], [10.0], [5.0], [10.0],
                [5.0], [10.0], [10.0], [5.0], [10.0]], dtype=torch.float32)
# x representa os valores de entrada (ex.: tempo ou quantidade de trabalho)

y = torch.tensor([[30.5],[63.0],[67.0],[29.0], [62.0],
                [30.5], [63.0], [67.0], [29.0], [62.0],
                [30.5], [63.0], [67.0], [29.0], [62.0],
                [30.5], [63.0], [67.0], [29.0], [62.0]], dtype=torch.float32)
# y representa os valores esperados de saída (ex.: tempo de conclusão)

# Define a classe da rede neural
class Net(nn.Module):  # Herda de nn.Module
    def __init__(self):
        super(Net, self).__init__()  # Inicializa a classe base
        self.fc1 = nn.Linear(1, 5)  # Primeira camada totalmente conectada (1 entrada, 5 saídas)
        self.fc2 = nn.Linear(5, 1)  # Segunda camada totalmente conectada (5 entradas, 1 saída)

    def forward(self, x):  # Define o fluxo de dados na rede
        x = torch.relu(self.fc1(x))  # Aplica a função de ativação ReLU na saída da primeira camada
        x = self.fc2(x)  # Passa pela segunda camada
        return x  # Retorna a saída final

# Instancia o modelo da rede neural
model = Net()

# Define a função de perda (MSE - Mean Squared Error)
criterion = nn.MSELoss()

# Define o otimizador (SGD - Stochastic Gradient Descent) com taxa de aprendizado de 0.01
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Loop de treinamento
for epoch in range(1000):  # Executa 1000 épocas
    optimizer.zero_grad()  # Zera os gradientes acumulados
    outputs = model(x)  # Faz a previsão com os dados de entrada
    loss = criterion(outputs, y)  # Calcula a perda entre a previsão e os valores esperados
    loss.backward()  # Calcula os gradientes
    optimizer.step()  # Atualiza os pesos do modelo com base nos gradientes

    # Exibe a perda a cada 100 épocas
    if epoch % 100 == 99:
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Faz uma previsão com o modelo treinado
with torch.no_grad():  # Desativa o cálculo de gradientes (modo de avaliação)
    predicted = model(torch.tensor([[10.0]], dtype=torch.float32))  # Faz a previsão para o valor 10.0
    print((f'Previsao de tempo de conclusao: {predicted.item()} minutos'))