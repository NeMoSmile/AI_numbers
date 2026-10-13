import numpy as np
from dataset import get_mnist
from network import NeuralNetwork

def train_and_evaluate():
    X_train, y_train, X_test, y_test = get_mnist()
    
    # Создаем сеть
    nn = NeuralNetwork(input_size=784, hidden_size=300, output_size=10, lr=0.1)
    
    epochs = 10
    batch_size = 64
    num_samples = X_train.shape[0]
    
    print("\n--- Начало обучения (NumPy) ---")
    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        epoch_loss = 0.0
        batches = num_samples // batch_size
        
        for i in range(0, num_samples, batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            
            predictions = nn.forward(X_batch)
            loss = nn.compute_loss(y_batch, predictions)
            nn.backward(X_batch, y_batch, predictions)
            
            epoch_loss += loss
            
        print(f"Эпоха [{epoch+1}/{epochs}] | Ошибка (Loss): {epoch_loss / batches:.4f}")

    print("\n--- Тестирование модели ---")
    test_predictions = nn.forward(X_test)
    predicted_classes = np.argmax(test_predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_test) * 100
    print(f"Точность на тесте: {accuracy:.2f}%")

    # Автоматическое сохранение прогресса
    nn.save_weights("weights.npz")

if __name__ == "__main__":
    train_and_evaluate()
