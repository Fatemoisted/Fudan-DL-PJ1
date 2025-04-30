from abc import abstractmethod
import numpy as np

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass


class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=np.random.randn, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(in_dim, out_dim) * np.sqrt(2.0 / in_dim)
        self.b = initialize_method(1, out_dim) * np.sqrt(2.0 / out_dim)
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        out = np.dot(X, self.params['W']) + self.params['b']
        return out

    def backward(self, grad : np.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        batch_size = grad.shape[0]
        
        # Calculate gradient for W
        dW = np.dot(self.input.T, grad)
        
        # Add weight decay gradient if enabled
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.params['W']
            
        # Calculate gradient for b (sum across the batch dimension)
        db = np.sum(grad, axis=0, keepdims=True)
        
        # Calculate gradient to pass to the previous layer
        dX = np.dot(grad, self.params['W'].T)
        
        self.grads['W'] = dW
        self.grads['b'] = db
        
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}

class conv2D(Layer):
    """
    The 2D convolutional layer. Try to implement it on your own.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, initialize_method=np.random.randn, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize kernels and bias
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        
        fan_in = in_channels * kernel_size[0] * kernel_size[1]
        self.W = initialize_method(out_channels, in_channels, kernel_size[0], kernel_size[1]) * np.sqrt(2.0 / fan_in)
        self.b = initialize_method(out_channels,) * np.sqrt(2.0 / out_channels)
        
        self.grads = {'W': None, 'b': None}
        self.input = None
        
        self.params = {'W': self.W, 'b': self.b}
        
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    # def forward(self, X):
    #     """
    #     input X: [batch, channels, H, W]
    #     W : [out_channels, in_channels, k, k]
    #     """
    #     self.input = X
    #     batch_size, _, H, W = X.shape
    #     k_h, k_w = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        
    #     # Calculate output dimensions
    #     out_h = (H + 2 * self.padding - k_h) // self.stride + 1
    #     out_w = (W + 2 * self.padding - k_w) // self.stride + 1
        
    #     # Initialize output
    #     out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        
    #     # Pad input if necessary
    #     if self.padding > 0:
    #         X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
    #     else:
    #         X_padded = X
        
    #     # Perform convolution
    #     for b in range(batch_size):
    #         for c_out in range(self.out_channels):
    #             for h in range(out_h):
    #                 for w in range(out_w):
    #                     h_start = h * self.stride
    #                     w_start = w * self.stride
    #                     h_end = h_start + k_h
    #                     w_end = w_start + k_w
                        
    #                     # Extract patch and perform convolution
    #                     patch = X_padded[b, :, h_start:h_end, w_start:w_end]
    #                     out[b, c_out, h, w] = np.sum(patch * self.params['W'][c_out]) + self.params['b'][c_out]
        
    #     return out
    def forward(self, X):
        """
        input X: [batch, channels, H, W]
        W : [out_channels, in_channels, k, k]
        """
        self.input = X
        batch_size, in_channels, H, W = X.shape
        k_h, k_w = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        
        # 计算输出维度
        out_h = (H + 2 * self.padding - k_h) // self.stride + 1
        out_w = (W + 2 * self.padding - k_w) // self.stride + 1
        
        # 填充输入
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            X_padded = X
        
        # 完全向量化的im2col实现
        cols = np.zeros((batch_size, in_channels * k_h * k_w, out_h * out_w))
        
        for i in range(k_h):
            i_max = i + self.stride * out_h
            for j in range(k_w):
                j_max = j + self.stride * out_w
                for c in range(in_channels):
                    # 提取所有适当偏移的元素
                    field = X_padded[:, c, i:i_max:self.stride, j:j_max:self.stride]
                    # 将其放入cols数组的相应位置
                    cols[:, c * k_h * k_w + i * k_w + j, :] = field.reshape(batch_size, -1)
        
        # 重塑权重并计算卷积
        W_reshaped = self.params['W'].reshape(self.out_channels, -1)  # [out_channels, in_channels*k_h*k_w]
        
        # 批量矩阵乘法
        out = np.matmul(W_reshaped, cols) + self.params['b'].reshape(self.out_channels, 1)
        
        # 重塑输出
        out = out.reshape(batch_size, self.out_channels, out_h, out_w)
        
        return out

    # def backward(self, grads):
    #     """
    #     grads : [batch_size, out_channel, new_H, new_W]
    #     """
    #     batch_size, _, H, W = self.input.shape
    #     _, _, out_h, out_w = grads.shape
    #     k_h, k_w = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        
    #     # Initialize gradients
    #     dW = np.zeros_like(self.W)
    #     db = np.zeros_like(self.b)
    #     dX = np.zeros_like(self.input)
        
    #     # Pad input if necessary
    #     if self.padding > 0:
    #         X_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
    #         dX_padded = np.zeros_like(X_padded)
    #     else:
    #         X_padded = self.input
    #         dX_padded = np.zeros_like(X_padded)
        
    #     # Calculate gradients
    #     for b in range(batch_size):
    #         for c_out in range(self.out_channels):
    #             for h in range(out_h):
    #                 for w in range(out_w):
    #                     h_start = h * self.stride
    #                     w_start = w * self.stride
    #                     h_end = h_start + k_h
    #                     w_end = w_start + k_w
                        
    #                     # Gradient for weights
    #                     dW[c_out] += X_padded[b, :, h_start:h_end, w_start:w_end] * grads[b, c_out, h, w]
                        
    #                     # Gradient for biases
    #                     db[c_out] += grads[b, c_out, h, w]
                        
    #                     # Gradient for input
    #                     dX_padded[b, :, h_start:h_end, w_start:w_end] += self.params['W'][c_out] * grads[b, c_out, h, w]
        
    #     # Add weight decay if enabled
    #     if self.weight_decay:
    #         dW += self.weight_decay_lambda * self.params['W']
        
    #     # Remove padding from dX_padded to get dX
    #     if self.padding > 0:
    #         dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
    #     else:
    #         dX = dX_padded
        
    #     self.grads['W'] = dW
    #     self.grads['b'] = db
        
    #     return dX

    def backward(self, grads):
        """
        grads : [batch_size, out_channel, new_H, new_W]
        """
        batch_size, in_channels, H, W = self.input.shape
        k_h, k_w = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        _, _, out_h, out_w = grads.shape
        
        # 填充输入
        if self.padding > 0:
            X_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            X_padded = self.input
        
        # 初始化梯度
        dW = np.zeros_like(self.params['W'])
        db = np.sum(grads, axis=(0, 2, 3))  # 直接计算偏置梯度
        
        # 使用im2col构建输入矩阵
        cols = np.zeros((batch_size, in_channels * k_h * k_w, out_h * out_w))
        
        # 更高效的im2col实现
        for i in range(k_h):
            i_max = i + self.stride * out_h
            for j in range(k_w):
                j_max = j + self.stride * out_w
                for c in range(in_channels):
                    field = X_padded[:, c, i:i_max:self.stride, j:j_max:self.stride]
                    cols[:, c * k_h * k_w + i * k_w + j, :] = field.reshape(batch_size, -1)
        
        # 重塑梯度为矩阵乘法的形式
        grads_reshaped = grads.reshape(batch_size, self.out_channels, -1)  # [batch, out_channels, out_h*out_w]
        
        # 批量计算权重梯度
        for b in range(batch_size):
            dW_b = np.matmul(grads_reshaped[b], cols[b].T).reshape(self.out_channels, in_channels, k_h, k_w)
            dW += dW_b
        
        # 计算输入梯度
        W_reshaped = self.params['W'].reshape(self.out_channels, -1)  # [out_channels, in_channels*k_h*k_w]
        
        # 批量计算输入梯度的列形式
        dX_cols = np.zeros((batch_size, in_channels * k_h * k_w, out_h * out_w))
        for b in range(batch_size):
            dX_cols[b] = np.matmul(W_reshaped.T, grads_reshaped[b])
        
        # 将dX_cols转回图像形式（逆im2col）
        dX_padded = np.zeros_like(X_padded)
        
        # 高效的逆im2col实现
        for i in range(k_h):
            i_max = i + self.stride * out_h
            for j in range(k_w):
                j_max = j + self.stride * out_w
                for c in range(in_channels):
                    idx = c * k_h * k_w + i * k_w + j
                    # 从dX_cols中获取适当的切片，并将其重塑回原始形状
                    field_grad = dX_cols[:, idx, :].reshape(batch_size, out_h, out_w)
                    # 使用高级索引将梯度添加到dX_padded的正确位置
                    for b in range(batch_size):
                        dX_padded[b, c, i:i_max:self.stride, j:j_max:self.stride] += field_grad[b]
        
        # 添加权重衰减
        if self.weight_decay:
            dW += self.weight_decay_lambda * self.params['W']
        
        # 去除填充
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
        
        self.grads['W'] = dW
        self.grads['b'] = db
        
        return dX
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}
        
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = np.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = np.where(self.input < 0, 0, grads)
        return output

class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model=None, max_classes=10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.has_softmax = True
        self.softmax_output = None
        self.labels = None
        self.batch_size = None
        self.grads = None
        
        self.optimizable = False

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D]
        labels : [batch_size, ]
        This function generates the loss.
        """
        self.batch_size = predicts.shape[0]
        self.labels = labels
        
        # Apply softmax if needed
        if self.has_softmax:
            self.softmax_output = softmax(predicts)
        else:
            self.softmax_output = predicts
        
        # Calculate cross-entropy loss
        log_likelihood = -np.log(self.softmax_output[np.arange(self.batch_size), labels])
        loss = np.sum(log_likelihood) / self.batch_size
        
        return loss
    
    def backward(self):
        # Compute gradients from loss to the input
        self.grads = self.softmax_output.copy()
        self.grads[np.arange(self.batch_size), self.labels] -= 1
        self.grads /= self.batch_size
        
        # Send the gradients to model for back propagation
        self.model.backward(self.grads)

    def cancel_soft_max(self):
        self.has_softmax = False
        return self
    
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    def __init__(self, model, lambda_reg=1e-4):
        super().__init__()
        self.model = model
        self.lambda_reg = lambda_reg
        self.optimizable = False
    
    def forward(self, loss):
        """
        Add L2 regularization term to the loss
        """
        reg_loss = 0
        for layer in self.model.layers:
            if hasattr(layer, 'params') and 'W' in layer.params:
                reg_loss += 0.5 * self.lambda_reg * np.sum(layer.params['W'] ** 2)
        
        return loss + reg_loss
    
    def backward(self, loss_grad=1.0):
        """
        Add regularization gradient to each layer's weight gradients
        """
        for layer in self.model.layers:
            if hasattr(layer, 'params') and 'W' in layer.params and hasattr(layer, 'grads'):
                layer.grads['W'] += self.lambda_reg * layer.params['W']
        
        return loss_grad
       
def softmax(X):
    X = X + 1e-7
    x_max = np.max(X, axis=1, keepdims=True)
    x_exp = np.exp(X - x_max)
    partition = np.sum(x_exp, axis=1, keepdims=True)
    return x_exp / partition

class AvgPool2D(Layer):
    """
    Average Pooling 2D layer that downsamples the input by taking the average
    value in each pooling window.
    """
    def __init__(self, pool_size, stride=None, padding=0) -> None:
        super().__init__()
        if isinstance(pool_size, int):
            self.pool_size = (pool_size, pool_size)
        else:
            self.pool_size = pool_size
            
        self.stride = stride if stride is not None else self.pool_size
        if isinstance(self.stride, int):
            self.stride = (self.stride, self.stride)
            
        self.padding = padding
        self.input = None
        self.input_shape = None
        
        self.optimizable = False
    
    def __call__(self, X) -> np.ndarray:
        return self.forward(X)
    
    def forward(self, X):
        """
        Perform average pooling on input X
        
        input X: [batch, channels, height, width]
        output: [batch, channels, new_height, new_width]
        """
        self.input = X
        self.input_shape = X.shape
        
        batch_size, channels, height, width = X.shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride
        
        # Calculate output dimensions
        out_height = (height + 2 * self.padding - pool_height) // stride_height + 1
        out_width = (width + 2 * self.padding - pool_width) // stride_width + 1
        
        # Pad input if necessary
        if self.padding > 0:
            X_padded = np.pad(X, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), 'constant')
        else:
            X_padded = X
        
        # Initialize output
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        # Perform average pooling
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * stride_height
                        w_start = w * stride_width
                        h_end = h_start + pool_height
                        w_end = w_start + pool_width
                        
                        pool_region = X_padded[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, h, w] = np.mean(pool_region)
        
        return output
    
    def backward(self, grads):
        """
        Compute gradients for average pooling layer
        
        input grads: [batch, channels, out_height, out_width]
        output: [batch, channels, height, width]
        """
        batch_size, channels, out_height, out_width = grads.shape
        _, _, height, width = self.input_shape
        pool_height, pool_width = self.pool_size
        stride_height, stride_width = self.stride
        
        # Initialize gradients for input
        dX = np.zeros(self.input_shape)
        
        # If there's padding in the forward pass, we need to account for it
        if self.padding > 0:
            padded_shape = (batch_size, channels, 
                            height + 2 * self.padding, 
                            width + 2 * self.padding)
            dX_padded = np.zeros(padded_shape)
        else:
            dX_padded = dX
        
        # Distribute gradients
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * stride_height
                        w_start = w * stride_width
                        h_end = h_start + pool_height
                        w_end = w_start + pool_width
                        
                        # Calculate gradient for each element in the pooling window
                        # For average pooling, gradient is evenly distributed to all elements
                        pool_size = pool_height * pool_width
                        dX_padded[b, c, h_start:h_end, w_start:w_end] += grads[b, c, h, w] / pool_size
        
        # Remove padding if necessary to get the original shape
        if self.padding > 0:
            dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dX = dX_padded
        
        return dX

class Dropout(Layer):
    """
    Dropout层：在训练时随机将一部分输入元素置为0
    """
    def __init__(self, dropout_ratio=0.5):
        super().__init__()
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.training = True  # 标记是否处于训练模式
        self.params = {}
        self.grads = {}
        
    def __call__(self, x):
        return self.forward(x)
    
    def forward(self, x):
        if self.training:
            # 生成与输入同形状的随机掩码
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            # 缩放保持期望值不变
            return x * self.mask / (1.0 - self.dropout_ratio)
        else:
            # 测试时不使用dropout
            return x
    
    def backward(self, dout):
        # 反向传播时，只对未被丢弃的神经元传递梯度
        if self.training:
            return dout * self.mask / (1.0 - self.dropout_ratio)
        else:
            return dout
            
    def clear_grad(self):
        # Dropout层没有需要优化的参数，所以这个方法是空的
        pass
    
    def train(self):
        """设置为训练模式"""
        self.training = True
        
    def eval(self):
        """设置为评估模式"""
        self.training = False