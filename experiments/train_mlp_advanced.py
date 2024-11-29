from _context import vugrad
import numpy as np
from argparse import ArgumentParser
import vugrad as vg
from vugrad.core import Op, TensorNode

class ReLU(Op):
    @staticmethod
    def forward(context, input):
        context['mask'] = input > 0
        return np.maximum(input, 0)

    @staticmethod
    def backward(context, goutput):
        return goutput * context['mask']

class DeepMLP(vg.Module):
    def __init__(self, input_size, output_size, hidden_sizes=[512, 256, 128], 
                 activation='relu', residual=False, init_type='glorot'):
        super().__init__()
        
        self.layers = []
        self.activation = activation
        self.residual = residual
        
        # Input layer
        prev_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layer = vg.Linear(prev_size, hidden_size)
            
            if init_type == 'zeros':
                layer.parameters()[0].value = np.zeros_like(layer.parameters()[0].value)  # weights
                layer.parameters()[1].value = np.zeros_like(layer.parameters()[1].value)  # bias
            elif init_type == 'normal':
                layer.parameters()[0].value = np.random.normal(0, 0.01, layer.parameters()[0].value.shape)
                layer.parameters()[1].value = np.zeros_like(layer.parameters()[1].value)
            
            self.layers.append(layer)
            prev_size = hidden_size
        
        # Output layer
        self.output_layer = vg.Linear(prev_size, output_size)

    def forward(self, input):
        x = input
        residual_connections = []  # Stack to store residual connections
        
        
        for i, layer in enumerate(self.layers):
            # Store input for residual connection
            if self.residual and i % 2 == 0:
                residual_connections.append(x)
            
            x = layer(x)
            
            if self.activation == 'relu':
                x = ReLU.do_forward(x)
            else:
                x = vg.sigmoid(x)
            
            # Add residual connection every 2 layers
            if self.residual and i % 2 == 1 and residual_connections:
                residual = residual_connections.pop()
                if x.size() == residual.size():  # Check dimensions match
                    x = x + residual
        
        x = self.output_layer(x)
        return vg.logsoftmax(x)

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        params.extend(self.output_layer.parameters())
        return params

def train_with_momentum(model, xtrain, ytrain, xval, yval, epochs=20, 
                       batch_size=128, lr=0.0001, momentum=0.9):
    n = xtrain.shape[0]
    accuracies = []
    losses = []
    
    # Initialize momentum buffers
    velocity = {i: np.zeros_like(param.value) for i, param in enumerate(model.parameters())}
    
    for epoch in range(epochs):
        # Validation accuracy
        o = model(vg.TensorNode(xval))
        predictions = np.argmax(o.value, axis=1)
        acc = (predictions == yval).sum() / yval.shape[0]
        accuracies.append(acc)
        o.clear()
        
        # Training
        epoch_loss = 0.0
        num_batches = 0
        
        for fr in range(0, n, batch_size):
            to = min(fr + batch_size, n)
            batch, targets = xtrain[fr:to, :], ytrain[fr:to]
            
            batch = vg.TensorNode(value=batch)
            outputs = model(batch)
            loss = vg.logceloss(outputs, targets)
            
            # Accumulate loss
            epoch_loss += loss.value
            num_batches += 1
            
            loss.backward()
            
            # Update with momentum
            for i, param in enumerate(model.parameters()):
                velocity[i] = momentum * velocity[i] - lr * param.grad
                param.value += velocity[i]
            
            loss.zero_grad()
            loss.clear()
        
        # Calculate average loss for the epoch
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        
        print(f'epoch {epoch:03} - training loss: {avg_loss:.4f} - val accuracy: {acc:.4f}')
    
    return accuracies, losses

def main():
    parser = ArgumentParser()
    parser.add_argument('-D', '--dataset', dest='data', default='mnist', type=str)
    parser.add_argument('-b', '--batch-size', dest='batch_size', default=128, type=int)
    parser.add_argument('-e', '--epochs', dest='epochs', default=20, type=int)
    parser.add_argument('-l', '--learning-rate', dest='lr', default=0.00001, type=float)
    parser.add_argument('-m', '--momentum', dest='momentum', default=0.9, type=float)
    args = parser.parse_args()

    # Load data
    (xtrain, ytrain), (xval, yval), num_classes = vg.load_mnist(final=False, flatten=True)
    num_features = xtrain.shape[1]

    # Test different configurations
    configs = [
        {'name': 'Baseline', 'hidden_sizes': [128], 'residual': False, 'init_type': 'glorot'},
        {'name': 'Deep', 'hidden_sizes': [128, 128, 128], 'residual': False, 'init_type': 'glorot'},
        {'name': 'Deep+Residual', 'hidden_sizes': [128, 128, 128, 128], 'residual': True, 'init_type': 'glorot'},
        {'name': 'Deep+Zero-Init', 'hidden_sizes': [128, 128], 'residual': False, 'init_type': 'zeros'},
        {'name': 'Deep+Normal-Init', 'hidden_sizes': [128, 128, 128], 'residual': False, 'init_type': 'normal'}
    ]

    results = {}
    for config in configs:
        print(f"\nTraining {config['name']}...")
        model = DeepMLP(
            input_size=num_features, 
            output_size=num_classes,
            hidden_sizes=config['hidden_sizes'],
            residual=config['residual'],
            init_type=config['init_type']
        )
        
        accuracies, losses = train_with_momentum(
            model, xtrain, ytrain, xval, yval,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            momentum=args.momentum
        )
        
        results[config['name']] = {
            'final_accuracy': accuracies[-1],
            'final_loss': losses[-1]
        }

    print("\nFinal Results:")
    for name, metrics in results.items():
        print(f"{name}:")
        print(f"  Train Loss: {metrics['final_loss']:.4f}")
        print(f"  Val Accuracy: {metrics['final_accuracy']:.4f}")

if __name__ == "__main__":
    main() 