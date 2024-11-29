from _context import vugrad
import numpy as np
from argparse import ArgumentParser
import vugrad as vg
from vugrad.core import Op, TensorNode

# Add ReLU class
class ReLU(Op):
    """
    Op for element-wise ReLU (Rectified Linear Unit) activation
    """
    @staticmethod
    def forward(context, input):
        context['mask'] = input > 0
        return np.maximum(input, 0)

    @staticmethod
    def backward(context, goutput):
        return goutput * context['mask']

# Modified MLP class
class MLP(vg.Module):
    def __init__(self, input_size, output_size, hidden_mult=4, activation='sigmoid'):
        super().__init__()
        
        hidden_size = hidden_mult * input_size
        self.layer1 = vg.Linear(input_size, hidden_size)
        self.layer2 = vg.Linear(hidden_size, output_size)
        self.activation = activation

    def forward(self, input):
        assert len(input.size()) == 2

        hidden = self.layer1(input)

        if self.activation == 'relu':
            hidden = ReLU.do_forward(hidden)
        else:
            hidden = vg.sigmoid(hidden)

        output = self.layer2(hidden)
        output = vg.logsoftmax(output)

        return output

    def parameters(self):
        return self.layer1.parameters() + self.layer2.parameters()

# Parse command line arguments
parser = ArgumentParser()

parser.add_argument('-D', '--dataset',
                dest='data',
                help='Which dataset to use. [synth, mnist]',
                default='mnist', type=str)

parser.add_argument('-b', '--batch-size',
                dest='batch_size',
                help='The batch size.',
                default=128, type=int)

parser.add_argument('-e', '--epochs',
                dest='epochs',
                help='The number of epochs.',
                default=20, type=int)

parser.add_argument('-l', '--learning-rate',
                dest='lr',
                help='The learning rate.',
                default=0.0001, type=float)

args = parser.parse_args()

def train_mlp(activation='sigmoid', epochs=args.epochs, batch_size=args.batch_size, 
              lr=args.lr, dataset=args.data):
    # Load data
    if dataset == 'synth':
        (xtrain, ytrain), (xval, yval), num_classes = vg.load_synth()
    else:
        (xtrain, ytrain), (xval, yval), num_classes = vg.load_mnist(final=False, flatten=True)
    
    num_instances, num_features = xtrain.shape
    
    # Create model
    mlp = MLP(input_size=num_features, output_size=num_classes, activation=activation)
    
    n, b = xtrain.shape[0], batch_size
    accuracies = []

    print(f'\n## Starting training with {activation} activation')
    for epoch in range(epochs):
        # Validation accuracy
        o = mlp(vg.TensorNode(xval))
        oval = o.value
        predictions = np.argmax(oval, axis=1)
        acc = (predictions == yval).sum() / yval.shape[0]
        accuracies.append(acc)
        
        print(f'epoch {epoch:03} accuracy: {acc:.4}')
        
        o.clear()
        
        # Training
        for fr in range(0, n, b):
            to = min(fr + b, n)
            batch, targets = xtrain[fr:to, :], ytrain[fr:to]
            
            batch = vg.TensorNode(value=batch)
            outputs = mlp(batch)
            loss = vg.logceloss(outputs, targets)
            
            loss.backward()
            
            for parm in mlp.parameters():
                parm.value -= lr * parm.grad
            
            loss.zero_grad()
            loss.clear()
    
    return accuracies

# Train both versions
sigmoid_acc = train_mlp(activation='sigmoid')
relu_acc = train_mlp(activation='relu')

print("\nFinal accuracies:")
print(f"Sigmoid: {sigmoid_acc[-1]:.4f}")
print(f"ReLU: {relu_acc[-1]:.4f}") 