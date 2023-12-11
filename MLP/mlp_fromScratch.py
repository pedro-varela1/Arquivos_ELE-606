import numpy as np

class Activation:
    def relu(self, x):
        return np.maximum(0, x)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        exp_vals = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_vals / np.sum(exp_vals, axis=-1, keepdims=True)

class MLP(Activation):
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Inicialização de pesos e bias para a camada oculta e camada de saída
        self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, output_size))

    def forward(self, x):
        # Camada oculta
        self.hidden_layer_input = np.dot(x, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = self.relu(self.hidden_layer_input)

        # Camada de saída
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        output = self.softmax(self.output_layer_input)

        return output

    def backward(self, x, y, output):
        # Backpropagation

        # Erro na camada de saída
        error_output = output - y

        # Gradiente da camada de saída
        grad_weights_output = np.dot(self.hidden_layer_output.T, error_output)
        grad_bias_output = np.sum(error_output, axis=0, keepdims=True)

        # Atualização dos pesos e bias da camada de saída
        self.weights_output -= self.learning_rate * grad_weights_output
        self.bias_output -= self.learning_rate * grad_bias_output

        # Erro na camada oculta
        error_hidden = np.dot(error_output, self.weights_output.T)
        error_hidden[self.hidden_layer_input <= 0] = 0  # Aplicação da derivada da função de ativação relu

        # Gradiente da camada oculta
        grad_weights_hidden = np.dot(x.T, error_hidden)
        grad_bias_hidden = np.sum(error_hidden, axis=0, keepdims=True)

        # Atualização dos pesos e bias da camada oculta
        self.weights_hidden -= self.learning_rate * grad_weights_hidden
        self.bias_hidden -= self.learning_rate * grad_bias_hidden

    def train(self, x_train, y_train, epochs):
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(x_train)

            # Backward pass
            self.backward(x_train, y_train, output)

            # Cálculo da perda (loss)
            loss = -np.sum(y_train * np.log(output))

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

# Exemplo de uso da rede neural para treinamento
input_size = 4    # Tamanho do vetor de entrada
hidden_size = 3   # Quantd. de neurons na camada escondida
output_size = 2   # Tamanho do vetor de output
learning_rate = 0.1   # Taxa de Aprendizado
epochs = 1000   # Epochs de Treinamento

# Dados de entrada e saída (exemplo)
x_train = np.random.randn(10, input_size)
y_train = np.array([[1, 0] if np.sum(data) > 0 else [0, 1] for data in x_train])

# Criando a rede neural
model = MLP(input_size, hidden_size, output_size, learning_rate)

# Treinamento da rede neural
model.train(x_train, y_train, epochs)

# Checando predictions
print(f'\n Valores de Saida Esperados:\n {y_train}')
predict = model.forward(x_train)
print(f'\n Valores de Saida da Rede Neural:\n {np.round(predict, 2)}')
