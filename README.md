# OCR Project

This repository contains an Optical Character Recognition (OCR) system built using PyTorch. The project is designed to recognize text from the IAM Handwriting Dataset and includes two models for experimentation: a baseline model and a more advanced branched model. Each model is trained using the IAM dataset.

## Models

### 1. Baseline Model

The BaselineModel is built on top of the EfficientNet_b1 backbone, which has been trimmed to improve performance for OCR tasks. It includes:

- **Backbone Modifications:** removed the last 30% of layers and freezes 50% of the remaining layers to reduce computational overhead. Strides are reduced for finer-grained feature extraction.
- **LSTM Layer:** A bidirectional LSTM is used to capture temporal dependencies in the sequence of features, with 256 hidden units.
- **Dropout:** Dropout of 30% is applied to regularize the model.
- **Final Layer:** The output is passed through a fully connected layer and a LogSoftmax for classification.

### 2. Branched Model

The BranchedModel builds on top of the EfficientNet_b1 backbone but introduces multiple convolutional branches to capture multi-scale features.

- **Backbone:** The first five layers of EfficientNet_b1 are used, with custom strides applied to maximize feature extraction.
- **Multi-Branch Design:** Three convolutional branches are used, each with different configurations to capture features at multiple scales.
- **LSTM Layer:** A 3-layer bidirectional LSTM is used, with 256 hidden units.
- **Temporal Dropout:** Dropout is applied to temporal features for regularization.
- **Final Projector:** The concatenated features from the branches are passed through a multi-layer projector, with LogSoftmax applied for output.

## Dataset

The models are trained on the IAM Handwriting Database, which contains annotated handwriting samples.

### Data Preprocessing

The project includes scripts for preprocessing the dataset:

- **`download_data.py`**: Downloads the dataset from Google Drive and extracts it into the appropriate format.
- **`process_data.py`**: Converts images to grayscale, resizes them to a standard size, and generates token mappings for labels.
- **`dataset.py`**: Defines a PyTorch Dataset class for the IAM dataset and implements data pipline, including data loading, preprocessing, and transformations. The data pipeline consists of transformations such as resizing and normalization applied to both training and testing sets. Additionally, augmentations like color jittering (brightness and contrast adjustments) are applied to the training set to enhance model robustness. The pipeline also handles data splitting into training and testing sets.

## Training

### Training Process

Both models use the CTC loss function for handling sequence-to-sequence training. The models are trained using the Adam optimizer with an adjustable learning rate and weight decay.

### Training Script

The `train.py` file contains the main training loop for both models:

- **Train Function:** Handles the training loop, updating model weights and calculating loss and accuracy metrics.
- **Test Function:** Evaluates the model on the test set and calculates the Character Error Rate (CER) and accuracy.
- **Save & Load Checkpoints:** Functionality for saving and loading model checkpoints is included.

### Arguments

The training script takes several command-line arguments:

- `--root_dir`: Directory containing the dataset.
- `--annotation_file`: File containing the image annotations.
- `--token_file`: JSON file mapping characters to indices.
- `--batch_size`: Batch size for training and testing.
- `--lr`: Learning rate for the optimizer.
- `--weight_decay`: Weight decay for regularization.
- `--model_type`: Choose between "baseline" and "branched" models.


## Note
For a detailed explanation on how to run and train the model using `argparse` from the command line, please refer to the attached notebook `training_colab.ipynb`.


## Evaluation

The project provides functionality to compute two key metrics during training and testing:

- **Character Error Rate (CER):** Used to evaluate how well the model transcribes handwritten text.
- **Accuracy:** Measures the overall performance of the model in predicting characters correctly.

## Requirements

To run this project, you need the following Python packages:

- torch
- torchvision
- numpy
- PIL
- tqdm
- gdown
- torchmetrics

You can install all dependencies via pip:

```bash
pip install torch torchvision numpy pillow tqdm gdown torchmetrics
