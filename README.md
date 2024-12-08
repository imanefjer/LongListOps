## Overview

This project involves building and training various neural network models for sequence classification tasks using PyTorch. The models include LSTM, LSTM with Attention, RNN, and AllenAI Longformer architectures, as well as a DistilBERT model for natural language processing tasks. The dataset used consists of expressions that are tokenized and converted into a format suitable for training.

### Task Description

Given the dataset, our goal is to determine the exact mathematical solution for expressions formatted in a tree-like structure. Each expression represents a mathematical operation, and the task is to evaluate these expressions to find the correct result. For example, the expression provided in the dataset can be evaluated to yield a specific numerical result.

## Example Data

The dataset used in this project includes expressions formatted like the one below:
```
[MAX 2 9 [MIN 4 7 ] 0 ]
```
**Note:** The actual expressions in the real dataset are much longer than this one.


## **Data Processing**
1. **Raw Input Sequences:** Sequences were initially provided as strings containing operations and numbers, e.g., `( ( [MIN 1 [MAX 2 3]] ) )`.
2. **Tokenization:**
   - A custom tokenizer was implemented to split sequences into tokens while handling nested operations.
   - Example:
     - Input: `[MIN 4 [MAX 2 9]]`
     - Tokens: `['[MIN', '4', '[MAX', '2', '9', ']', ']']`
3. **Tree Representation:** The tokenized sequences were converted into tree structures for hierarchical processing.
4. **Distribution Analysis:** Visualized input lengths before and after tokenization to understand the impact of processing on sequence length.



## Table of Contents
- [Overview](#overview)
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


## **Classes of Models**
### **1. RNN-Based Models**
- Implemented simple RNN and LSTM baselines.
- **Advantages:**
  - Sequential processing ensures all tokens are considered.
- **Challenges:**
  - Struggles with long-term dependencies.
  - Computationally expensive for long sequences.

### **2. Transformer-Based Models**
- **Transformer Classifier:** Implemented using standard Transformer architecture for sequence classification.
- **DistilBERT:** A pre-trained language model fine-tuned on ListOps data.
- **Advantages:**
  - Captures global relationships effectively.
  - Handles large-scale data with pre-trained embeddings.
- **Challenges:**
  - Quadratic attention complexity limits scalability for very long sequences.

### **3. Tree-Based Models**
- Explored Tree-LSTM and recursive models to handle hierarchical structures explicitly.
- **Advantages:**
  - Naturally suited for nested operations.
  - Processes data in a structured, interpretable manner.
- **Challenges:**
  - Computational overhead for deep or large trees.


---

## Model Architectures


### 1. RNN Model

The RNN model is implemented in `models/rnn_model.ipynb`. Similar to the LSTM model, it processes sequences but uses a simpler recurrent architecture. The model achieved an accuracy of **45.20%** on the test dataset.

### 2. LSTM Model

The LSTM model is implemented in `models/lstm_model.ipynb`. It utilizes PyTorch's `nn.LSTM` for sequence processing. The model achieved an accuracy of **59.50%** on the test dataset.
### 3. LSTM with Attention Model

The LSTM with Attention model is implemented in `models/lstm_with_attention_model.ipynb`. This model enhances the LSTM architecture by incorporating an attention mechanism, allowing it to focus on relevant parts of the input sequence. The model achieved an accuracy of **62.25%** on the test dataset.
### 4. AllenAI Longformer Model

The AllenAI Longformer model is implemented in `models/allenai Longformer.ipynb`. This model is designed for long document processing and utilizes attention mechanisms to handle longer sequences efficiently. The model achieved an accuracy of **17.25%** on the test dataset.
### 5. DistilBERT Model

The DistilBERT model is implemented in three versions: `ilyas_models/distilbert_train_5000.ipynb`, `ilyas_models/distilbert_train_30000.ipynb`, and `ilyas_models/distilbert_train_full.ipynb`. Each one was fine-tuned on 5000, 30000, and on the full train dataset, respectively. It leverages the transformer architecture for sequence classification tasks, providing state-of-the-art performance on various NLP tasks. The model that was fine-tuned on the full dataset achieved an accuracy of **43.75%** on the test dataset.
### 6. Transformer Classifier Model

The Transformer Classifier model is implemented in `models/TransformerClassifier.ipynb`. This model leverages transformer architecture for sequence classification tasks. It is designed to handle tokenized expressions and classify them based on the learned representations. The model achieved an accuracy of **38.65%** on the test dataset.

## Training and Evaluation

Each model is trained using a defined number of epochs, and the training process includes validation to monitor performance. Early stopping is implemented to prevent overfitting.


---

## **Relevant Links**
- [ListOps Dataset Paper](https://arxiv.org/abs/1906.04341)
- [Hugging Face Transformers Library](https://huggingface.co/transformers/)
- [Tree-LSTM Paper](https://arxiv.org/abs/1503.00075)
- [Longformer Paper](https://arxiv.org/abs/2004.05150)
- [BigBird Paper](https://arxiv.org/abs/2007.14062)

---

This repository serves as a foundation for exploring hierarchical sequence processing tasks. Contributions and suggestions for improvement are welcome!
