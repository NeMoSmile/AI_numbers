import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=300, output_size=10, lr=0.01):
        self.lr = lr
        
        # Инициализация Хе (He Initialization) для весов
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def softmax(self, Z):
        # Экспонента со сдвигом во избежание переполнения (Overflow)
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    def backward(self, X, y_true, y_pred):
        m = X.shape[0]  # Размер батча
        
        # One-hot encoding для истинных меток классов
        one_hot_y = np.zeros_like(y_pred)
        one_hot_y[np.arange(m), y_true] = 1.0
        
        # Градиент на выходном слое (Cross-Entropy + Softmax)
        dZ2 = y_pred - one_hot_y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Градиент на скрытом слое
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.relu_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        # Обновление параметров градиентным спуском (SGD)
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        # Избегаем логарифма от нуля с помощью клиппинга
        p = np.clip(y_pred[np.arange(m), y_true], 1e-15, 1.0)
        return -np.sum(np.log(p)) / m
    
    def save_weights(self, filepath="weights.npz"):
        np.savez(filepath, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
        print(f"Веса успешно сохранены в файл: {filepath}")

    def load_weights(self, filepath="weights.npz"):
        try:
            data = np.load(filepath)
            self.W1 = data['W1']
            self.b1 = data['b1']
            self.W2 = data['W2']
            self.b2 = data['b2']
            print(f"Веса успешно загружены из файла: {filepath}")
            return True
        except FileNotFoundError:
            print(f"Файл весов {filepath} не найден. Сеть будет использовать случайные веса.")
            return False