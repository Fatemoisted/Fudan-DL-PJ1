from .op import *
import pickle

class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None, dropout=0.0):
        self.size_list = size_list
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)
                    self.layers.append(Dropout(dropout))

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)
        self.size_list = param_list[0]
        self.act_func = param_list[1]

        for i in range(len(self.size_list) - 1):
            self.layers = []
            for i in range(len(self.size_list) - 1):
                layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
                layer.W = param_list[i + 2]['W']
                layer.b = param_list[i + 2]['b']
                layer.params['W'] = layer.W
                layer.params['b'] = layer.b
                layer.weight_decay = param_list[i + 2]['weight_decay']
                layer.weight_decay_lambda = param_list[i+2]['lambda']
                if self.act_func == 'Logistic':
                    raise NotImplemented
                elif self.act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                if i < len(self.size_list) - 2:
                    self.layers.append(layer_f)
        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable and not isinstance(layer, Dropout):
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
    
    def eval(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.train = False
    def train(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.train = True

        

class Model_CNN(Layer):
    """
    A model with conv2D layers. Implement it using the operators you have written in op.py
    """
    def __init__(self, arch=None, act_func='ReLU', weight_decay=False, lambda_val=1e-8, dropout = 0):
        """
        Initialize CNN model
        
        arch: List of layer specifications
            For conv layers: ('conv', in_channels, out_channels, kernel_size, stride, padding)
            For linear layers: ('linear', in_dim, out_dim)
            For flatten layer: ('flatten',)
        act_func: Activation function to use
        weight_decay: Whether to use weight decay
        lambda_val: Weight decay lambda value
        """
        super().__init__()
        self.arch = arch
        self.act_func = act_func
        self.weight_decay = weight_decay
        self.lambda_val = lambda_val
        
        if arch is not None:
            self.layers = []
            for layer_spec in arch:
                if layer_spec[0] == 'conv':
                    _, in_channels, out_channels, kernel_size, stride, padding = layer_spec
                    conv_layer = conv2D(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        padding=padding,
                        weight_decay=weight_decay,
                        weight_decay_lambda=lambda_val
                    )
                    self.layers.append(conv_layer)
                    
                    # Add activation after each conv layer
                    if act_func == 'ReLU':
                        self.layers.append(ReLU())
                        self.layers.append(Dropout(dropout))
                    else:
                        raise NotImplementedError(f"Activation function {act_func} not implemented")
                
                elif layer_spec[0] == 'flatten':
                    # Flatten layer doesn't need parameters, just reshape operation in forward
                    self.layers.append(Flatten())
                
                elif layer_spec[0] == 'linear':
                    _, in_dim, out_dim = layer_spec
                    linear_layer = Linear(
                        in_dim=in_dim,
                        out_dim=out_dim,
                        weight_decay=weight_decay,
                        weight_decay_lambda=lambda_val
                    )
                    self.layers.append(linear_layer)
                    
                    # Only add activation if it's not the last layer
                    if layer_spec != arch[-1] and act_func == 'ReLU':
                        self.layers.append(ReLU())
                
                else:
                    raise ValueError(f"Unknown layer type: {layer_spec[0]}")

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        """
        Forward pass through the CNN
        
        X: Input data with shape [batch_size, channels, height, width]
        """
        assert self.arch is not None, 'Model has not been initialized yet. Use model.load_model to load a model or create a new model with architecture specifications.'
        
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        
        return outputs

    def backward(self, loss_grad):
        """
        Backward pass through the CNN
        
        loss_grad: Gradient from the loss function
        """
        grads = loss_grad
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        
        return grads
    
    def load_model(self, param_path):
        """
        Load model parameters from a file
        
        param_path: Path to the parameter file
        """
        with open(param_path, 'rb') as f:
            param_list = pickle.load(f)
        
        self.arch = param_list[0]
        self.act_func = param_list[1]
        
        # Create layers based on architecture
        self.layers = []
        param_idx = 2  # Start after arch and act_func
        
        for layer_spec in self.arch:
            if layer_spec[0] == 'conv':
                _, in_channels, out_channels, kernel_size, stride, padding = layer_spec
                conv_layer = conv2D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding
                )
                
                # Load saved parameters
                conv_layer.W = param_list[param_idx]['W']
                conv_layer.b = param_list[param_idx]['b']
                conv_layer.params['W'] = conv_layer.W
                conv_layer.params['b'] = conv_layer.b
                conv_layer.weight_decay = param_list[param_idx]['weight_decay']
                conv_layer.weight_decay_lambda = param_list[param_idx]['lambda']
                
                self.layers.append(conv_layer)
                param_idx += 1
                
                # Add activation after each conv layer
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                else:
                    raise NotImplementedError(f"Activation function {self.act_func} not implemented")
            
            elif layer_spec[0] == 'flatten':
                self.layers.append(Flatten())
            
            elif layer_spec[0] == 'linear':
                _, in_dim, out_dim = layer_spec
                linear_layer = Linear(in_dim=in_dim, out_dim=out_dim)
                
                # Load saved parameters
                linear_layer.W = param_list[param_idx]['W']
                linear_layer.b = param_list[param_idx]['b']
                linear_layer.params['W'] = linear_layer.W
                linear_layer.params['b'] = linear_layer.b
                linear_layer.weight_decay = param_list[param_idx]['weight_decay']
                linear_layer.weight_decay_lambda = param_list[param_idx]['lambda']
                
                self.layers.append(linear_layer)
                param_idx += 1
                
                # Only add activation if it's not the last layer
                if layer_spec != self.arch[-1] and self.act_func == 'ReLU':
                    self.layers.append(ReLU())
            
            else:
                raise ValueError(f"Unknown layer type: {layer_spec[0]}")
        
    def save_model(self, save_path):
        """
        Save model parameters to a file
        
        save_path: Path to save the parameters
        """
        param_list = [self.arch, self.act_func]
        
        for layer in self.layers:
            if layer.optimizable and hasattr(layer, 'params') and not isinstance(layer, Dropout):
                param_dict = {
                    'W': layer.params['W'],
                    'b': layer.params['b'],
                    'weight_decay': layer.weight_decay,
                    'lambda': layer.weight_decay_lambda
                }
                param_list.append(param_dict)
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
    
    def eval(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.train = False
    def train(self):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.train = True

class Flatten(Layer):
    """
    A layer that flattens the input from [batch_size, channels, height, width]
    to [batch_size, channels*height*width]
    """
    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.optimizable = False
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        X: Input tensor with shape [batch_size, channels, height, width]
        """
        self.input_shape = X.shape
        batch_size = X.shape[0]
        flattened_size = np.prod(X.shape[1:])
        return X.reshape(batch_size, flattened_size)
    
    def backward(self, grad):
        """
        grad: Gradient from next layer with shape [batch_size, flattened_size]
        """
        return grad.reshape(self.input_shape)