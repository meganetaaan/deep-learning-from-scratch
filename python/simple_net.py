#%%
import sys, os
sys.path.append('./deep-learning-from-scratch')
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class SimpleNet:
  def __init__(self):
    self.W = np.random.randn(2, 3) # initialize with gaussian distribution

  def predict(self, x):
    return np.dot(x, self.W)

  def loss(self, x, t):
    z = self.predict(x)
    y = softmax(z)
    loss = cross_entropy_error(y, t)

    return loss

# %%
if __name__ == '__main__':
  net = SimpleNet()
  print(net.W)

  x = np.array([0.6, 0.9])
  p = net.predict(x)
  print(p)
  print(np.argmax(p))

  t = np.array([0, 0, 1])
  print(net.loss(x, t))

  f = lambda w: net.loss(x, t)
  dW = numerical_gradient(f, net.W)
  print(dW)
