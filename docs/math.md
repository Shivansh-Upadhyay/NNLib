# The math behind nnlib

Shapes: batch size N, layer input size `in`, output size `out`.

## Dense layer
```
Z = X * W + b          X: N x in    W: in x out    b: 1 x out    Z: N x out
A = f(Z)               f = ReLU, sigmoid, tanh, or nothing
```

## Backward pass
Given dA (N x out), how much the loss changes when this layer's output changes:
```
dZ = dA * f'(Z)        element by element
dW = Xt * dZ           Xt = X transposed
db = sum of dZ over the N rows
dX = dZ * Wt           handed to the layer before this one
```
Slopes: ReLU: 1 if z > 0 else 0. Sigmoid: s(1 - s). Tanh: 1 - tanh^2. None: 1.

## Losses
```
MSE:            L = mean((pred - target)^2)       dpred = 2 (pred - target) / count
Softmax:        p = exp(z - max) / sum(exp(z - max))   (subtracting the max avoids overflow)
Cross-entropy:  L = -(1/N) * sum(y * log(p))      y is one-hot
                dZ = (p - y) / N                  (softmax and cross-entropy together)
```

## Starting weights
```
He (ReLU):        normal numbers, mean 0, std = sqrt(2 / in)
Xavier (others):  uniform between -L and +L, L = sqrt(6 / (in + out))
```

## Optimizers
```
SGD:       w = w - lr * g
Momentum:  v = 0.9 v + g              w = w - lr * v
Adam:      m = 0.9 m + 0.1 g
           v = 0.999 v + 0.001 g^2
           m_hat = m / (1 - 0.9^t)    v_hat = v / (1 - 0.999^t)
           w = w - lr * m_hat / (sqrt(v_hat) + 1e-8)
```

## How we checked the gradients
For one number w: (loss(w + h) - loss(w - h)) / 2h should match the gradient from backward().
The tests do this for every activation, every loss, and a whole 2-layer network.
