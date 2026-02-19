import numpy as np
class TwoLayerMLP:
    def __init__(self, input_size, hidden_size, output_size, activation='relu', learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_type = activation
        self.lr = learning_rate
        

        scale = np.sqrt(2. / input_size) if activation == 'relu' else np.sqrt(1. / input_size)
        self.W1 = np.random.randn(input_size, hidden_size) * scale
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, hidden_size) * scale
        self.b2 = np.zeros((1, hidden_size))
        self.W3 = np.random.randn(hidden_size, output_size) * scale
        self.b3 = np.zeros((1, output_size))


    def activation(self, Z):
        if self.activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-Z))
        elif self.activation_type == 'relu':
            return np.maximum(0, Z)
            
    def activation_derivative(self, Z):
        if self.activation_type == 'sigmoid':
            s = 1 / (1 + np.exp(-Z))
            return s * (1 - s)
        elif self.activation_type == 'relu':
            return (Z > 0).astype(float)

    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.activation(self.Z1)
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.activation(self.Z2)
        
        self.Z3 = np.dot(self.A2, self.W3) + self.b3
        self.A3 = self.softmax(self.Z3)
        
        return self.A3

    def compute_loss(self, y_true_one_hot, y_pred):
        m = y_true_one_hot.shape[0]
        y_pred = np.clip(y_pred, 1e-12, 1. - 1e-12)
        loss = -np.sum(y_true_one_hot * np.log(y_pred)) / m
        return loss

    def backward(self, X, y_true_one_hot):
        m = X.shape[0]
        
    
        dZ3 = self.A3 - y_true_one_hot  
        dW3 = np.dot(self.A2.T, dZ3) / m
        db3 = np.sum(dZ3, axis=0, keepdims=True) / m
        
        dA2 = np.dot(dZ3, self.W3.T)
        dZ2 = dA2 * self.activation_derivative(self.Z2)
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * self.activation_derivative(self.Z1)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m
        
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

    def train(self, X_train, y_train, X_val, y_val, epochs=200):
        train_acc_history = []
        val_acc_history = []
        
        grad_w1_history = []
        grad_w2_history = []
        
        for i in range(epochs):
            y_pred = self.forward(X_train)

            loss = self.compute_loss(y_train, y_pred)
            
            m = X_train.shape[0]

            dZ3 = self.A3 - y_train
            dW3 = np.dot(self.A2.T, dZ3) / m
            db3 = np.sum(dZ3, axis=0, keepdims=True) / m
            
            dA2 = np.dot(dZ3, self.W3.T)
            dZ2 = dA2 * self.activation_derivative(self.Z2)
            dW2 = np.dot(self.A1.T, dZ2) / m
            db2 = np.sum(dZ2, axis=0, keepdims=True) / m
            
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.activation_derivative(self.Z1)
            dW1 = np.dot(X_train.T, dZ1) / m
            db1 = np.sum(dZ1, axis=0, keepdims=True) / m
            
            grad_w1_history.append(np.mean(np.abs(dW1)))
            grad_w2_history.append(np.mean(np.abs(dW2)))
            
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W3 -= self.lr * dW3
            self.b3 -= self.lr * db3
            
            train_acc = np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_train, axis=1))
            val_pred = self.forward(X_val)
            val_acc = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_val, axis=1))
            
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            
            if i % 20 == 0:
                print(f"Epoch {i}: Loss {loss:.4f} | Val Acc: {val_acc:.4f}")
                
        return train_acc_history, val_acc_history, grad_w1_history, grad_w2_history