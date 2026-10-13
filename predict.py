import sys
import os
import numpy as np
from PIL import Image, ImageOps
from network import NeuralNetwork

def predict_image(image_path):
    if not os.path.exists(image_path):
        print(f"Ошибка: Файл '{image_path}' не найден!")
        return

    # 1. Загружаем веса в нейросеть
    nn = NeuralNetwork(input_size=784, hidden_size=300, output_size=10)
    if not nn.load_weights("weights.npz"):
        print("Сначала запустите train.py, чтобы обучить сеть и сохранить веса!")
        return

    # 2. Предобработка картинки под формат MNIST
    # Открываем изображение и переводим в оттенки серого ('L')
    img = Image.open(image_path).convert('L')
    
    # Сжимаем/расширяем до 28x28 пикселей
    img = img.resize((28, 28), Image.Resampling.LANCZOS)
    
    # Инвертируем цвета: MNIST — это БЕЛЫЕ цифры на ЧЕРНОМ фоне.
    # Если ты нарисовал черную цифру на белом фоне, ImageOps.invert сделает её правильной для сети.
    img = ImageOps.invert(img)
    
    # Переводим в массив NumPy, нормализуем значения в диапазон [0.0, 1.0]
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # Выпрямляем матрицу 28x28 в вектор 1x784
    img_vector = img_array.reshape(1, 784)

    # 3. Прямой проход (Распознавание)
    probabilities = nn.forward(img_vector)
    predicted_digit = np.argmax(probabilities)
    confidence = probabilities[0][predicted_digit] * 100

    print(f"\nРезультат распознавания для '{image_path}':")
    print(f"Нейросеть считает, что это цифра: {predicted_digit} (Уверенность: {confidence:.2f}%)")
    
    # Выведем вероятности для всех цифр ради интереса
    print("\nРаспределение уверенности по всем классам:")
    for digit, prob in enumerate(probabilities[0]):
        print(f"Цифра {digit}: {prob * 100:.2f}%")

if __name__ == "__main__":
    # Можно передать путь к картинке через аргумент командной строки,
    # либо скрипт по умолчанию будет искать файл "digit.png"
    target_image = sys.argv[1] if len(sys.argv) > 1 else "pictures/digit.png"
    predict_image(target_image)
