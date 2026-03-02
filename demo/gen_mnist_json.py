# gen_mnist_json.py (20 Prototypes)
import json
import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.cluster import KMeans

print("Loading MNIST data...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ★強化点: 5 -> 20 (合計200パターン)
CLUSTERS_PER_DIGIT = 20 

train_data = []

print(f"Extracting {CLUSTERS_PER_DIGIT} prototypes per digit...")

for digit in range(10):
    indices = np.where(y_train == digit)[0]
    # 全データを使うと重いので2000枚から抽出
    images = x_train[indices[:2000]].reshape(-1, 784) 
    
    kmeans = KMeans(n_clusters=CLUSTERS_PER_DIGIT, n_init=5, random_state=42)
    kmeans.fit(images)
    centroids = kmeans.cluster_centers_
    
    for center in centroids:
        pixels = center.tolist()
        train_data.append({
            "__type": "Item",
            "label": str(digit), 
            "pixels": pixels
        })
    print(f"Digit {digit}: Done.")

test_data = []
# テストデータも100枚に
for digit in range(10):
    indices = np.where(y_test == digit)[0][:10]
    for idx in indices:
        pixels = x_test[idx].flatten().tolist()
        test_data.append({
            "__type": "Item",
            "label": str(digit),
            "pixels": pixels
        })

data = { "__type": "Dataset", "train": train_data, "test": test_data }

with open("mnist_data.json", "w") as f:
    json.dump(data, f)

print("mnist_data.json updated with 200 prototypes!")