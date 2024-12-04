import torch
from torch.utils.data import Dataset, DataLoader
import re

def tokenize(expression):
    """Convert expression string to tokens, preserving operators."""
    # Replace parentheses with spaces
    expr = expression.replace('(', ' ').replace(')', ' ')
    
    # Add spaces around brackets that aren't part of operators
    expr = re.sub(r'\[(?!(MIN|MAX|MED|SM))', ' [ ', expr)
    expr = expr.replace(']', ' ] ')
    
    # Split and filter empty strings
    return [token for token in expr.split() if token]

class ListOpsDataset(Dataset):
    def __init__(self, X, y, max_length=2000):
        """
        Args:
            X: Array of source expressions
            y: Array of target values
            max_length: Maximum sequence length (will pad/truncate to this)
        """
        self.X = X
        self.y = y
        self.max_length = max_length
        
        # Create vocabulary from operators and digits
        self.vocab = {
            'PAD': 0,  # Padding token
            '[MIN': 1,
            '[MAX': 2,
            '[MED': 3,
            '[SM': 4,
            ']': 5,
            '(': 6,
            ')': 7
        }
        # Add digits 0-9
        for i in range(10):
            self.vocab[str(i)] = i + 8
            
    def __len__(self):
        return len(self.X)
    
    def tokenize(self, expr):
        """Convert expression to token IDs."""
        tokens = tokenize(expr)  # Using our previous tokenize function
        return [self.vocab.get(token, 0) for token in tokens]
    
    def __getitem__(self, idx):
        expr = self.X[idx]
        target = self.y[idx]
        
        # Convert to token IDs
        token_ids = self.tokenize(expr)
        
        # Pad or truncate to max_length
        if len(token_ids) > self.max_length:
            token_ids = token_ids[:self.max_length]
        else:
            token_ids = token_ids + [self.vocab['PAD']] * (self.max_length - len(token_ids))
        
        return {
            'input_ids': torch.tensor(token_ids, dtype=torch.long),
            'target': torch.tensor(target, dtype=torch.long)
        }
