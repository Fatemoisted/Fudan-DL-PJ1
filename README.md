# MNIST Classification with MLP and CNN

This repository hosts a course project for building a mini deep-learning framework (`mynn`) from scratch and using it to train multilayer perceptron (MLP) and convolutional neural network (CNN) models on the [MNIST handwritten digit dataset](http://yann.lecun.com/exdb/mnist/). Besides the training scripts, the project also contains tools for data augmentation, visualization, and hyper-parameter exploration.

## Table of Contents
- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Dataset & Pretrained Weights](#dataset--pretrained-weights)
- [Training a Model](#training-a-model)
- [Evaluating a Saved Model](#evaluating-a-saved-model)
- [Utilities & Visualizations](#utilities--visualizations)
- [Tips & Troubleshooting](#tips--troubleshooting)

## Project Structure
```
Fudan-DL-PJ1/
├── best_models/               # Serialized parameters saved by RunnerM
├── cnn_visualizations/        # Example figures produced by weight_visualization.py
├── dataset_explore.ipynb      # Notebook for inspecting MNIST statistics
├── data_aug.py                # Data augmentation helpers (rotation, shift, elastic, ...)
├── draw_tools/                # Helper routines for plotting training curves
├── figs/                      # Example filter visualizations
├── hyperparameter_search.py   # Placeholder for automated sweeps
├── mynn/                      # Lightweight NN framework (ops, layers, optimizer, runner)
├── test_train.py              # End-to-end training script (MLP or CNN)
├── test_model.py              # Standalone evaluation script
└── weight_visualization.py    # Activation/filter visualization script
```

## Environment Setup
1. **Python**: Recommended Python 3.9+.
2. **Create a virtual environment** (optional but encouraged):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. **Install dependencies**:
   ```bash
   pip install numpy matplotlib tqdm scipy scikit-image
   ```
   The code only relies on standard scientific Python libraries. Install Jupyter if you plan to run the exploratory notebook.

## Dataset & Pretrained Weights
1. Download the MNIST gzip files and the pretrained model checkpoints from the shared Google Drive folder: <https://drive.google.com/drive/folders/1nGkhW3MdGlDmZ0MYguxIlGbxBM17zsHz?usp=sharing>
2. Create the following directory structure and place the files accordingly:
   ```
   dataset/
     MNIST/
       train-images-idx3-ubyte.gz
       train-labels-idx1-ubyte.gz
       t10k-images-idx3-ubyte.gz
       t10k-labels-idx1-ubyte.gz
   best_models/
     best_model_cnn.pickle      # Example model distributed with the project
   ```
3. If you train new models, they will be saved inside `best_models/` (or another directory you pass to `RunnerM.train`).

## Training a Model
`test_train.py` demonstrates the full pipeline: loading MNIST, optional data augmentation, model initialization, and training/validation/testing.

```bash
python test_train.py
```

Key configuration options live in the `config` dictionary:
- `model_type`: `"linear"` for an MLP or `"cnn"` for the convolutional architecture defined near the bottom of the script.
- `n_hidden`, `n_layer`: control the depth and width of the MLP.
- `kernel`, `padding`: convolution hyper-parameters for the CNN.
- `lambda_reg`, `dropout`, `lr`, `num_epochs`: regularization and optimization settings.
- `use_aug`: toggles the augmentation pipeline defined in `data_aug.py` (requires `scipy` and `scikit-image`).

During training the script:
1. Loads the raw gzip files with `numpy` and `gzip`.
2. Splits 10,000 samples as a validation set and caches the permutation index (`idx.pickle`).
3. Normalizes images with the training-set mean and standard deviation.
4. Runs `RunnerM.train`, which reports accuracy/loss every `log_iters` iterations and stores the best weights.

Feel free to copy `test_train.py` as a template for custom experiments (different architectures, optimizers, schedulers, etc.).

## Evaluating a Saved Model
Use `test_model.py` to run inference with a serialized CNN model:
```bash
python test_model.py
```
The script:
1. Loads `best_models/best_model_cnn.pickle` via `Model_CNN.load_model`.
2. Normalizes the test images with the same statistics used during training (update the constants if you retrain).
3. Iterates over the test set in mini-batches and prints the final accuracy.

To evaluate an MLP checkpoint, replace the `Model_CNN` definition with `Model_MLP` and point to the corresponding `.pickle` file.

## Utilities & Visualizations
- **`data_aug.py`**: Implements random rotations, translations, noise injection, zoom, and elastic deformation. Import `augment_data` inside your own training script to expand the dataset.
- **`weight_visualization.py`**: Loads a trained CNN and produces filter/activation visualizations (outputs saved under `cnn_visualizations/`). You can adapt this script to debug internal representations.
- **`draw_tools/plot.py`**: Helper for plotting loss/accuracy curves from a `RunnerM` instance.
- **`dataset_explore.ipynb`**: Optional notebook for sanity-checking pixel distributions, label balance, etc.
- **`hyperparameter_search.py`**: Stub file for grid search or random search utilities. Fill it with your preferred automation when running multiple experiments.

## Tips & Troubleshooting
- Set `np.random.seed(...)` before data loading to keep train/validation splits reproducible.
- GPU acceleration is not used; the `mynn` package performs all operations with NumPy, so expect longer training times compared with PyTorch/TF.
- Checkpoint files are simple pickled parameter dictionaries. Use `Model_MLP.save_model` / `Model_CNN.save_model` to export your own models.
- For faster experimentation, reduce the dataset size by subsampling `train_imgs` before calling `RunnerM.train`.
- If you encounter numerical issues, verify that input normalization is applied consistently to validation/test data.

Feel free to extend the repository with new layers, optimizers, or visual diagnostics. Contributions that make the `mynn` mini-framework more modular or improve experiment automation are especially welcome!
