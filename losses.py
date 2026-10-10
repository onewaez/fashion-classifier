import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.saved_probabilities = None

    def forward(self, logits, correct_class_indices):
        # Softmax with max subtraction for numerical stability
        max_logits = np.max(logits, axis=1, keepdims=True)
        safe_logits = logits - max_logits
        exponents = np.exp(safe_logits)
        
        self.saved_probabilities = exponents / np.sum(exponents, axis=1, keepdims=True)
        
        batch_size = logits.shape[0]
        correct_probs = self.saved_probabilities[np.arange(batch_size), correct_class_indices]
        
        log_probs = np.log(correct_probs + 1e-9)
        loss = -np.sum(log_probs) / batch_size
        
        return loss

    def backward(self, logits, correct_class_indices):
        batch_size = logits.shape[0]
        start_gradient = self.saved_probabilities.copy()
        start_gradient[np.arange(batch_size), correct_class_indices] -= 1.0
        
        return start_gradient / batch_size