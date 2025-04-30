import mynn as nn
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import os

save_dir = 'cnn_visualizations'
os.makedirs(save_dir, exist_ok=True)

model = nn.models.Model_CNN(arch=[
        ('conv', 1, 2, 5, 1, 2), ('conv', 2, 2, 5, 1, 2), ('flatten',), ('linear', 2*28*28, 10)
        ], weight_decay=True, dropout=0)
model.load_model(r'best_models\best_model_cnn.pickle')
print(model.layers)

conv1_weights = model.layers[0].params['W']
conv2_weights = model.layers[2].params['W']


plt.figure(figsize=(10, 5))
for i in range(conv1_weights.shape[0]):
    plt.subplot(1, conv1_weights.shape[0], i+1)
    kernel = conv1_weights[i, 0] 
    plt.imshow(kernel, cmap='viridis')
    plt.title(f'Filter {i+1}')
    plt.axis('off')
plt.suptitle('First Convolutional Layer Filters')
plt.tight_layout()
plt.savefig(f'{save_dir}/conv1_filters.png', dpi=300)
plt.show()


plt.figure(figsize=(15, 8))
for i in range(conv2_weights.shape[0]):
    for j in range(conv2_weights.shape[1]):
        plt.subplot(conv2_weights.shape[0], conv2_weights.shape[1], i*conv2_weights.shape[1]+j+1)
        kernel = conv2_weights[i, j]
        plt.imshow(kernel, cmap='viridis')
        plt.title(f'Filter {i+1}, Channel {j+1}')
        plt.axis('off')
plt.suptitle('Second Convolutional Layer Filters')
plt.tight_layout()
plt.savefig(f'{save_dir}/conv2_filters.png', dpi=300)
plt.show()


test_images_path = r'.\dataset\MNIST\t10k-images-idx3-ubyte.gz'
test_labels_path = r'.\dataset\MNIST\t10k-labels-idx1-ubyte.gz'

with gzip.open(test_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        X=np.frombuffer(f.read(), dtype=np.uint8).reshape(num, 1, 28, 28)
    
with gzip.open(test_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        y = np.frombuffer(f.read(), dtype=np.uint8)

X = (X - 33.32747961734694) / 78.57427268218198


def visualize_predictions(model, X, y, num_samples=10):
    indices = np.random.choice(len(X), num_samples, replace=False)
    
    plt.figure(figsize=(15, 8))
    for i, idx in enumerate(indices):
        img = X[idx, 0]
        true_label = y[idx]

        pred_logits = model(X[idx:idx+1])
        pred_label = np.argmax(pred_logits)
        
        plt.subplot(2, 5, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {true_label}, Pred: {pred_label}')
        plt.axis('off')
    
    plt.suptitle('Model Predictions on Test Set')
    plt.tight_layout()
    plt.savefig(f'{save_dir}/model_predictions.png', dpi=300)
    plt.show()


def get_samples_by_class(X, y, samples_per_class=1):
    samples = {}
    for i in range(len(X)):
        label = y[i]
        if label not in samples or len(samples.get(label, [])) < samples_per_class:
            if label not in samples:
                samples[label] = []
            samples[label].append(i)
        if len(samples) == 10 and all(len(indices) == samples_per_class for indices in samples.values()):
            break
    return samples


def visualize_multiple_activations(model, X, y, num_digits=5):
    samples_dict = get_samples_by_class(X, y, samples_per_class=1)
    selected_digits = sorted(list(samples_dict.keys()))[:num_digits]
    sample_indices = [samples_dict[digit][0] for digit in selected_digits]
    
    all_activations_conv1 = []
    all_activations_conv2 = []
    all_original_images = []
    all_labels = []
    
    for idx in sample_indices:
        image = X[idx]
        label = y[idx]

        all_original_images.append(image[0])
        all_labels.append(label)
        
        x = image.reshape(1, 1, 28, 28)
        
        z1 = model.layers[0].forward(x)
        all_activations_conv1.append(z1[0]) 

        z2 = model.layers[1].forward(z1)
        all_activations_conv2.append(z2[0]) 
    
    fig = plt.figure(figsize=(20, 5 * num_digits))

    gs = fig.add_gridspec(num_digits, 1 + 2 + 2) 
    
    for i in range(num_digits):
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(all_original_images[i], cmap='gray')
        ax_img.set_title(f'Digit {all_labels[i]}')
        ax_img.axis('off')
        
        for j in range(2):
            ax_conv1 = fig.add_subplot(gs[i, j+1])
            ax_conv1.imshow(all_activations_conv1[i][j], cmap='viridis')
            ax_conv1.set_title(f'Conv1 Ch{j+1}')
            ax_conv1.axis('off')
        
        for j in range(2):
            ax_conv2 = fig.add_subplot(gs[i, j+3])
            ax_conv2.imshow(all_activations_conv2[i][j], cmap='viridis')
            ax_conv2.set_title(f'Conv2 Ch{j+1}')
            ax_conv2.axis('off')
    
    plt.suptitle('Sample Activations Across Network Layers', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/sample_activations_combined.png', dpi=300)
    plt.show()
    
    conv2_channels = all_activations_conv2[0].shape[0]
    
    if conv2_channels > 2:
        fig = plt.figure(figsize=(20, 5 * num_digits))
        gs = fig.add_gridspec(num_digits, conv2_channels)
        
        for i in range(num_digits):
            for j in range(conv2_channels):
                ax = fig.add_subplot(gs[i, j])
                ax.imshow(all_activations_conv2[i][j], cmap='viridis')
                if j == 0:
                    ax.set_ylabel(f'Digit {all_labels[i]}', fontsize=12)
                if i == 0:
                    ax.set_title(f'Channel {j+1}')
                ax.axis('off')
        
        plt.suptitle('Second Convolutional Layer - All Channels', fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{save_dir}/conv2_all_channels.png', dpi=300)
        plt.show()

def visualize_all_digits_activations(model, X, y):
    samples_dict = get_samples_by_class(X, y, samples_per_class=1)
    all_digits = sorted(samples_dict.keys())
    if len(all_digits) != 10:
        print(f"Warning: Not all digits are present. Found: {all_digits}")

    all_digits_conv1 = {digit: None for digit in all_digits}

    for digit in all_digits:
        idx = samples_dict[digit][0]
        image = X[idx]
        x = image.reshape(1, 1, 28, 28)
        z1 = model.layers[0].forward(x)
        all_digits_conv1[digit] = z1[0]  
    
    fig = plt.figure(figsize=(15, 20))
    
    gs = fig.add_gridspec(10, 3)
    
    for i, digit in enumerate(all_digits):
        original_img = X[samples_dict[digit][0]][0]
        ax_img = fig.add_subplot(gs[i, 0])
        ax_img.imshow(original_img, cmap='gray')
        ax_img.set_title(f'Digit {digit}')
        ax_img.axis('off')

        for j in range(2):
            ax_conv1 = fig.add_subplot(gs[i, j+1])
            ax_conv1.imshow(all_digits_conv1[digit][j], cmap='viridis')
            ax_conv1.set_title(f'Conv1 Ch{j+1}')
            ax_conv1.axis('off')
    
    plt.suptitle('First Layer Activations for All Digits (0-9)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_digits_conv1_activations.png', dpi=300)
    plt.show()
    
    all_digits_conv2 = {digit: None for digit in all_digits}
    
    for digit in all_digits:
        idx = samples_dict[digit][0]
        image = X[idx]

        x = image.reshape(1, 1, 28, 28)

        z1 = model.layers[0].forward(x)

        z2 = model.layers[1].forward(z1)
        all_digits_conv2[digit] = z2[0]  

    conv2_channels = all_digits_conv2[all_digits[0]].shape[0]

    max_channels_to_show = min(6, conv2_channels) 
    
    fig = plt.figure(figsize=(3*max_channels_to_show, 2*10))
    gs = fig.add_gridspec(10, max_channels_to_show)
    
    for i, digit in enumerate(all_digits):
        for j in range(max_channels_to_show):
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(all_digits_conv2[digit][j], cmap='viridis')
            if j == 0:
                ax.set_ylabel(f'Digit {digit}', fontsize=12)
            if i == 0:
                ax.set_title(f'Channel {j+1}')
            ax.axis('off')
    
    plt.suptitle('Second Layer Activations for All Digits (0-9)', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/all_digits_conv2_activations.png', dpi=300)
    plt.show()

def create_activation_heatmaps(model, X, y):
    samples_dict = get_samples_by_class(X, y, samples_per_class=5)
    all_digits = sorted(samples_dict.keys())
    avg_conv1_activations = {digit: [] for digit in all_digits}
    avg_conv2_activations = {digit: [] for digit in all_digits}
    for digit in all_digits:
        conv1_sum = None
        conv2_sum = None
        count = 0
        
        for idx in samples_dict[digit]:
            image = X[idx]
            x = image.reshape(1, 1, 28, 28)
            z1 = model.layers[0].forward(x)
            z2 = model.layers[1].forward(z1)
            if conv1_sum is None:
                conv1_sum = z1[0].copy()
                conv2_sum = z2[0].copy()
            else:
                conv1_sum += z1[0]
                conv2_sum += z2[0]
            
            count += 1
        avg_conv1 = conv1_sum / count
        avg_conv2 = conv2_sum / count

        avg_conv1_activation = np.mean(avg_conv1, axis=(1, 2))
        avg_conv2_activation = np.mean(avg_conv2, axis=(1, 2))
        
        avg_conv1_activations[digit] = avg_conv1_activation
        avg_conv2_activations[digit] = avg_conv2_activation

    plt.figure(figsize=(12, 8))
    heatmap_data_conv1 = np.zeros((len(all_digits), 2)) 
    for i, digit in enumerate(all_digits):
        heatmap_data_conv1[i] = avg_conv1_activations[digit]
    
    plt.imshow(heatmap_data_conv1, cmap='hot')
    plt.colorbar(label='Average Activation Strength')
    plt.xlabel('Conv1 Channel')
    plt.ylabel('Digit')
    plt.title('Conv1 Layer Activation Strength by Digit')
    plt.xticks(np.arange(2), ['Channel 1', 'Channel 2'])
    plt.yticks(np.arange(10), all_digits)
    for i in range(10):
        for j in range(2):
            plt.text(j, i, f'{heatmap_data_conv1[i, j]:.2f}', 
                     ha="center", va="center", color="black" if heatmap_data_conv1[i, j] < 0.7 else "white")
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conv1_activation_heatmap.png', dpi=300)
    plt.show()

    plt.figure(figsize=(14, 8))
    conv2_channels = len(avg_conv2_activations[all_digits[0]])
    heatmap_data_conv2 = np.zeros((len(all_digits), conv2_channels))
    for i, digit in enumerate(all_digits):
        heatmap_data_conv2[i] = avg_conv2_activations[digit]
    
    plt.imshow(heatmap_data_conv2, cmap='hot')
    plt.colorbar(label='Average Activation Strength')
    plt.xlabel('Conv2 Channel')
    plt.ylabel('Digit')
    plt.title('Conv2 Layer Activation Strength by Digit')
    plt.xticks(np.arange(conv2_channels), [f'Ch {i+1}' for i in range(conv2_channels)])
    plt.yticks(np.arange(10), all_digits)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/conv2_activation_heatmap.png', dpi=300)
    plt.show()

random_indices = np.random.choice(len(X), min(100, len(X)), replace=False)
X_subset = X[random_indices]
y_subset = y[random_indices]

visualize_predictions(model, X_subset, y_subset, num_samples=10)
visualize_multiple_activations(model, X_subset, y_subset, num_digits=5)
visualize_all_digits_activations(model, X_subset, y_subset)
create_activation_heatmaps(model, X_subset, y_subset)