## Overview
# Project README

## Overview

This project involves building and training various neural network models for sequence classification tasks using PyTorch. The models include LSTM, LSTM with Attention, RNN, and AllenAI Longformer architectures, as well as a DistilBERT model for natural language processing tasks. The dataset used consists of expressions that are tokenized and converted into a format suitable for training.

### Task Description

Given the dataset, our goal is to determine the exact mathematical solution for expressions formatted in a tree-like structure. Each expression represents a mathematical operation, and the task is to evaluate these expressions to find the correct result. For example, the expression provided in the dataset can be evaluated to yield a specific numerical result.

## Example Data

The dataset used in this project includes expressions formatted in a tree-like structure. Below is an example of how the data appears :
```
[MAX
├── 6
├── [MED
│   ├── [MIN
│   │   ├── 2
│   │   └── 9
│   └── 3
└── 6
```
## Table of Contents

- [Project README](#project-readme)
  - [Overview](#overview-1)
    - [Task Description](#task-description)
  - [Example Data](#example-data)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Data Preparation](#data-preparation)
    - [Tokenization Function](#tokenization-function)
  - [Model Architectures](#model-architectures)
    - [RNN Model](#rnn-model)
    - [LSTM Model](#lstm-model)
    - [LSTM with Attention Model](#lstm-with-attention-model)
    - [AllenAI Longformer Model](#allenai-longformer-model)
    - [DistilBERT Model](#distilbert-model)
    - [Transformer Classifier Model](#transformer-classifier-model)
  - [Training and Evaluation](#training-and-evaluation)

## Installation

To set up the environment, ensure you have Python 3.9 or higher installed. You can create a virtual environment and install the required packages using the following commands:

```
pip install torch torchvision torchaudio
pip install transformers datasets tqdm matplotlib
```
## Data Preparation

The dataset consists of expressions that need to be tokenized. The `ListOpsDataset` class is defined in `data/ListOpsDataset.py` to handle the loading and processing of the dataset. The tokenization function converts expressions into tokens while preserving operators.

### Tokenization Function

```python
def tokenize(expression):
    """Convert expression string to tokens, preserving operators."""
    expr = expression.replace('(', ' ').replace(')', ' ')
    expr = expr.replace(']', ' ] ')
    return [token for token in expr.split() if token]
```


## Model Architectures


### RNN Model

The RNN model is implemented in `models/rnn_model.ipynb`. Similar to the LSTM model, it processes sequences but uses a simpler recurrent architecture. The model achieved an accuracy of **45.20%** on the test dataset.

### LSTM Model

The LSTM model is implemented in `models/lstm_model.ipynb`. It utilizes PyTorch's `nn.LSTM` for sequence processing. The model achieved an accuracy of **59.50%** on the test dataset.
### LSTM with Attention Model

The LSTM with Attention model is implemented in `models/lstm_with_attention_model.ipynb`. This model enhances the LSTM architecture by incorporating an attention mechanism, allowing it to focus on relevant parts of the input sequence. The model achieved an accuracy of **62.25%** on the test dataset.
### AllenAI Longformer Model

The AllenAI Longformer model is implemented in `models/allenai Longformer.ipynb`. This model is designed for long document processing and utilizes attention mechanisms to handle longer sequences efficiently. The model achieved an accuracy of **17.25%** on the test dataset.
### DistilBERT Model

The DistilBERT model is implemented in `ilyas_models/distilbert_train_5000.ipynb`. It leverages the transformer architecture for sequence classification tasks, providing state-of-the-art performance on various NLP tasks. The model achieved an accuracy of **38.35%** on the test dataset.
### Transformer Classifier Model

The Transformer Classifier model is implemented in `models/TransformerClassifier.ipynb`. This model leverages transformer architecture for sequence classification tasks. It is designed to handle tokenized expressions and classify them based on the learned representations. The model achieved an accuracy of **38.65%** on the test dataset.

## Training and Evaluation

Each model is trained using a defined number of epochs, and the training process includes validation to monitor performance. Early stopping is implemented to prevent overfitting.


