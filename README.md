# Hierarchical Sequence Processing with Neural Networks

## Overview

This repository explores hierarchical sequence processing using a variety of neural network models and architectures. Our core objective is to accurately evaluate and classify complex mathematical expressions, such as those found in the ListOps dataset. Each expression involves nested operations (e.g., `MIN`, `MAX`, `MED`, `SM`), and our task is to predict the correct integer result.

We experiment with a diverse set of models, including:

- **RNN-Based Models:** Bidirectional RNNs, Bidirectional LSTMs, and Bidirectional LSTMs with Attention  
- **Transformer-Based Models:**  
  - AllenAI Longformer for long-range dependency handling  
  - DistilBERT for leveraging pre-trained language representations  
  - Custom Transformer Classifier models for sequence classification tasks
- **Sparse-Attention Models:** BigBird, designed for efficient processing of very long sequences

This comprehensive approach allows us to compare performance, scalability, and suitability of different architectures for long-sequence tasks.

---

## Task Description

The ListOps dataset consists of synthetic, tree-like mathematical expressions. Each expression may look like:

```
[MAX 2 9 [MIN 4 7 ] 0 ]
```

Our goal is to evaluate this expression and return a single integer result (0-9). While the above example is short, the actual dataset can contain very long and deeply nested expressions, challenging traditional sequence models due to long-range dependencies and computational complexity.

---

## Example Data

A typical line from the dataset might look like:

```
Source:  [MED 7 [MIN 2 9] [MAX 3 4 [MIN 1 1]]]
Target:  3
```

The source is an expression with nested operations, and the target is the evaluated integer result.

---

## Data Processing

1. **Raw Input Sequences:**  
   Originally provided as strings of nested operations and numbers.

2. **Tokenization:**  
   A custom tokenizer extracts meaningful tokens while handling parentheses and brackets:
   ```python
   def tokenize(expression):
       expr = expression.replace('(', ' ').replace(')', ' ')
       expr = re.sub(r'\[(?!(MIN|MAX|MED|SM))', ' [ ', expr)
       expr = expr.replace(']', ' ] ')
       return [token for token in expr.split() if token]
   ```
   
   For example:
   ```
   Input:  [MIN 4 [MAX 2 9]]
   Output Tokens: ['[MIN', '4', '[MAX', '2', '9', ']', ']']
   ```

3. **Vocabulary Construction:**  
   The tokenized expressions are mapped to a custom vocabulary of operators and digits.

4. **Collation & Padding:**  
   We use PyTorch DataLoaders with custom `collate_fn` functions to batch, pad, and mask sequences efficiently.

---

## Installation

To set up your environment:

```bash
pip install torch torchvision torchaudio transformers datasets tqdm matplotlib
```

We recommend Python 3.9+ and a GPU-enabled environment (e.g., Google Colab) for faster training.

---

## Data Preparation

The dataset is stored as `.tsv` files with `Source` and `Target` columns. We provide a `ListOpsDataset` class that:

- Loads data from TSV files.
- Tokenizes expressions.
- Converts tokens to integer IDs.
- Provides PyTorch-compatible indexing and iteration.

**Snippet:**

```python
train_dataset = ListOpsDataset(X_train, y_train)
val_dataset = ListOpsDataset(X_val, y_val)
test_dataset = ListOpsDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, collate_fn=collate_fn)
```

---

## Model Architectures

### 1. Bidirectional RNN Model

- **File:** `models/bidirectional_rnn_model.ipynb`  
- **Description:** A Bidirectional RNN-based classifier. While straightforward, it struggles with very long sequences due to vanishing gradients and limited context.  
- **Performance:** ~45.20% test accuracy

### 2. Bidirectional LSTM Model

- **File:** `models/bidirectional_lstm_model.ipynb`  
- **Description:** Leverages Bidirectional LSTM units to better capture long-term dependencies.  
- **Performance:** ~59.50% test accuracy

### 3. LSTM with Attention Model

- **File:** `models/bidirectional_lstm_with_attention_model.ipynb`  
- **Description:** Adds an attention mechanism to the bidirectional LSTM, allowing the model to focus on the most relevant parts of the sequence.  
- **Performance:** ~62.25% test accuracy

### 4. AllenAI Longformer Model

- **File:** `models/allenai_longformer.ipynb`  
- **Description:** Uses the Longformer architecture from AllenAI, which employs sparse attention for long documents.  
- **Performance:** ~17.25% test accuracy (lower on this particular task, indicating complexity in tuning)

### 5. DistilBERT Model

- **Files:** `ilyas_models/distilbert_train_5000.ipynb`, `ilyas_models/distilbert_train_30000.ipynb`, `ilyas_models/distilbert_train_full.ipynb`  
- **Description:** Fine-tunes DistilBERT (a lighter version of BERT) on subsets or the full dataset.  
- **Performance:** ~43.75% test accuracy on the full training set

### 6. Transformer Classifier Model

- **File:** `models/TransformerClassifier.ipynb` (and similar variants)  
- **Description:** Implements a standard Transformer-based encoder to classify sequences. While powerful, the quadratic attention complexity can become expensive for very long inputs.  
- **Performance:** ~38.65% test accuracy

### 7. BigBird Model (Sparse Attention Transformer)

- **File:** Adapted code snippet in `models/bigbird_classifier.py`  
- **Description:** BigBird uses block-sparse attention for more scalable processing of long sequences. It can handle input lengths beyond what is feasible for standard Transformers.  
- **Key Idea:** By combining sparse and random attention, BigBird reduces the complexity from quadratic to more manageable scales, enabling it to process longer sequences efficiently.

**Code Snippet (Simplified):**

```python
from transformers import BigBirdConfig, BigBirdForSequenceClassification

config = BigBirdConfig(
    vocab_size=len(train_dataset.vocab),
    hidden_size=128,
    num_hidden_layers=2,
    num_attention_heads=2,
    intermediate_size=512,
    max_position_embeddings=512,
    num_labels=10,
    attention_type="block_sparse",
    block_size=64,
    num_random_blocks=1
)

model = BigBirdForSequenceClassification(config)
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Training loop (simplified)
for epoch in range(2):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        targets = batch['target'].to(device)
        attention_mask = (input_ids != train_dataset.vocab['PAD']).long()

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=targets)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

**Performance:** With appropriate tuning (layers, block sizes, training steps), BigBird aims to improve scalability and potentially accuracy on very long sequences.

---

## Training and Evaluation

We provide training loops for all models. Each loop involves:

1. **Forward Pass:** Computing predictions and loss.  
2. **Backward Pass:** Backpropagating gradients to update parameters.  
3. **Evaluation:** Periodically evaluating on validation and test sets to monitor overfitting and performance.

**General Training Template:**

```python
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
```

---

## Summary of Results

| Model                  | Test Accuracy | Notes                                |
|------------------------|---------------|---------------------------------------|
| Bidirectional RNN                    | 45.20%        | Baseline, struggles with long deps    |
| Bidirectional LSTM                   | 59.50%        | Better long-term dependencies         |
| Bidirectional LSTM + Attention       | 62.25%        | Focuses on important parts of input   |
| AllenAI Longformer     | 17.25%        | Requires more tuning for this dataset |
| DistilBERT (Full Data) | 43.75%        | Leverages pre-trained language model  |
| Transformer Classifier | 38.65%        | Good, but expensive for very long seq |
| BigBird                | TBD           | Promising for very long sequences     |

*(TBD: BigBird performance depends on tuning hyperparameters and training regime.)*

---

## Further Explanations

- **Why Hierarchical Expressions?**  
  Traditional sequence models excel at linear dependencies. Hierarchical expressions challenge models to handle nested structures and long-range dependencies, highlighting differences in model architectures.

- **Why Transformers and BigBird?**  
  Transformers have shown state-of-the-art results in NLP. BigBird extends these capabilities to handle extremely long sequences efficiently by using sparse attention patterns.

- **Trade-offs:**  
  - **Bidirectional RNNs/LSTMs:** Simpler but struggle with very long inputs.  
  - **Transformers:** Powerful but can be expensive (O(n²) complexity in sequence length).  
  - **Sparse-Attention Models (Longformer, BigBird):** More efficient for long sequences, but may require careful tuning and can be more complex to implement.

---

## Relevant Links

- [ListOps Dataset Paper](https://arxiv.org/abs/1906.04341)  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [Longformer Paper](https://arxiv.org/abs/2004.05150)  
- [BigBird Paper](https://arxiv.org/abs/2007.14062)  
- [Tree-LSTM Paper](https://arxiv.org/abs/1503.00075)

