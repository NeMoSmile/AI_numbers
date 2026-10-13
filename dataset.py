import os
import gzip
import urllib.request
import numpy as np

# Официальное зеркало датасета MNIST
URL = "https://googleapis.com"
FILES = {
    "train_img": "train-images-idx3-ubyte.gz",
    "train_lbl": "train-labels-idx1-ubyte.gz",
    "test_img": "t10k-images-idx3-ubyte.gz",
    "test_lbl": "t10k-labels-idx1-ubyte.gz"
}

def download_mnist(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    for name, filename in FILES.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Скачивание {filename}...")
            urllib.request.urlretrieve(URL + filename, filepath)
    print("Все файлы MNIST на месте.")

def load_images(filepath):
    with gzip.open(filepath, 'rb') as f:
        # Пропускаем первые 16 байт метаданных
        data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    # Нормализуем изображения: переводим пиксели из [0, 255] в [0.0, 1.0]
    return data.reshape(-1, 28 * 28).astype(np.float32) / 255.0

def load_labels(filepath):
    with gzip.open(filepath, 'rb') as f:
        # Пропускаем первые 8 байт метаданных
        return np.frombuffer(f.read(), dtype=np.uint8, offset=8)

def get_mnist():
    data_dir = "./data"
    download_mnist(data_dir)
    
    X_train = load_images(os.path.join(data_dir, FILES["train_img"]))
    y_train = load_labels(os.path.join(data_dir, FILES["train_lbl"]))
    X_test = load_images(os.path.join(data_dir, FILES["test_img"]))
    y_test = load_labels(os.path.join(data_dir, FILES["test_lbl"]))
    
    return X_train, y_train, X_test, y_test
