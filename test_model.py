import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm

model = nn.models.Model_CNN(arch=[
        ('conv', 1, 2, 5, 1, 2), ('conv', 2, 2, 5, 1, 2), ('flatten',), ('linear', 2*28*28, 10)
        ], weight_decay=True, dropout=0)
model.load_model(r'best_models\best_model_cnn.pickle')

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        X=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)

X = (X - 33.32747961734694) / 78.57427268218198


n_samples = X.shape[0]
total_score = 0.0
n_batches = int(np.ceil(n_samples / 32))

for batch_idx in tqdm(range(n_batches)):
        start_idx = batch_idx * 32
        end_idx = min((batch_idx + 1) * 32, n_samples)
        batch_X = X[start_idx:end_idx]
        batch_y = y[start_idx:end_idx]

        batch_logits = model(batch_X)
        batch_score = nn.metric.accuracy(batch_logits, batch_y)

        batch_weight = (end_idx - start_idx) / n_samples
        total_score += batch_score * batch_weight
print(f"Accuracy: {total_score:.4f}")