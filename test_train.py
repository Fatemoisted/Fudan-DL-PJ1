# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot

import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from data_aug import augment_data
# fixed seed for experiment
np.random.seed(309)

config = {
    'num_epochs': 5,
    'log_iters': 100,
    "model_type": "cnn", # linear, cnn
    "n_hidden": 512,
    "n_layer": 3,
    "lambda_reg": 1e-6,
    "lr": 1e-1,
    "dropout": 0.0,
    "kernel": 5,
    "padding": 2,
    "use_aug": False
}

train_images_path = r'.\dataset\MNIST\train-images-idx3-ubyte.gz'
train_labels_path = r'.\dataset\MNIST\train-labels-idx1-ubyte.gz'

test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        if config["model_type"] == "cnn":
            train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)
        else:
            train_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = np.frombuffer(f.read(), dtype=np.uint8)

idx = np.random.permutation(np.arange(num))

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        if config["model_type"] == "cnn":
            test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)
        else:
            test_imgs=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 28*28)
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        test_labs = np.frombuffer(f.read(), dtype=np.uint8)


# choose 10000 samples from train set as validation set.

# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

if config["use_aug"]:
        aug_train_imgs, aug_train_labs = augment_data(train_imgs, train_labs, num_augmented_samples=20000)
        train_imgs = np.concatenate([train_imgs, aug_train_imgs], axis=0)
        train_labs = np.concatenate([train_labs, aug_train_labs], axis=0)
        shuffle_idx = np.random.permutation(len(train_imgs))
        train_imgs = train_imgs[shuffle_idx]
        train_labs = train_labs[shuffle_idx]

mean = train_imgs.mean()
std = train_imgs.std()
import pdb; pdb.set_trace()
train_imgs = (train_imgs - mean) / std
valid_imgs = (valid_imgs - mean) / std
test_imgs = (test_imgs - mean) / std

# B, _ = train_imgs.shape()
# train_imgs = train_imgs.reshape(B, 1, 28, 28)
# valid_imgs = valid_imgs.reshape(B, 1, 28, 28)
if config["model_type"] == "cnn":
        model = nn.models.Model_CNN(arch=[
                ('conv', 1, 2, config["kernel"], 1, config["padding"]), ('conv', 2, 2, config["kernel"], 1, config["padding"]), ('flatten',), ('linear', 2*28*28, 10)
                ], lambda_val=config["lambda_reg"], weight_decay=True, dropout=config["dropout"])
else:
        model = nn.models.Model_MLP([train_imgs.shape[-1]]+[config["n_hidden"] for _ in range(config["n_layer"]-2)] + [10], 'ReLU', [config["lambda_reg"] for _ in range(config["n_layer"]-1)])
# optimizer = nn.optimizer.SGD(init_lr=0.06, model=cnn_model)
optimizer = nn.optimizer.MomentGD(init_lr=config["lr"], model=model)
scheduler = nn.lr_scheduler.Constant(optimizer=optimizer)
loss_fn = nn.op.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)

runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=config["num_epochs"], log_iters=300, save_dir=r'./best_models')

model.load_model(r'best_models\best_model.pickle')
runner = nn.runner.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, scheduler=scheduler)
runner.model.eval()
total_score, total_loss = runner.evaluate([test_imgs, test_labs])
print(f"[Test] loss: {total_loss}, score: {total_score}")

# _, axes = plt.subplots(1, 2)
# axes.reshape(-1)
# _.set_tight_layout(1)
# plot(runner, axes)

# plt.show()