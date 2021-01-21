#%%
import numpy as np
import sys, os
sys.path.append('./deep-learning-from-scratch')
from dataset.mnist import load_mnist
import matplotlib.pylab as plt

#%%
def mean_squared_error(y, t):
  return 0.5 * np.sum((y - t)**2)

def cross_entropy_error(y, t):
  if y.ndim == 1:
    t = t.reshape(1, t.size)
    y = y.reshape(1, y.size)

  # 教師データがone-hot-vectorの場合、正解ラベルのインデックスに変換
  if t.size == y.size:
    t = t.argmax(axis=1)

  batch_size = y.shape[0]
  return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

def numerical_diff(f, x):
  h = 1e-4 # 0.0001
  return (f(x + h) - f(x - h)) / (2*h)

def function_1(x):
  return 0.01*x**2 + 0.1*x

def function_2(x):
  return x[0]**2 + x[1]**2


def numerical_gradient(f, x):
  h = 1e-4 # 0.0001
  grad = np.zeros_like(x)

  it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
  while not it.finished:
    idx = it.multi_index
    tmp_val = x[idx]
    x[idx] = tmp_val + h
    fxh1 = f(x) # f(x+h)

    x[idx] = tmp_val - h
    fxh2 = f(x) # f(x-h)
    grad[idx] = (fxh1 - fxh2) / (2*h)

    x[idx] = tmp_val # 値を元に戻す
    it.iternext()

  return grad

def gradient_descent(f, init_x, lr=0.01, step_num=100):
  x = init_x,

  for i in range(step_num):
    grad = numerical_gradient(f, x)
    x -= lr * grad

  return x
#%%
if __name__ == '__main__':
  (x_train, t_train), (x_test, t_test) = \
    load_mnist(normalize=True, one_hot_label=True)

  print(x_train.shape)
  print(t_train.shape)

  train_size = x_train.shape[0]
  batch_size = 10
  batch_mask = np.random.choice(train_size, batch_size)
  x_batch = x_train[batch_mask]
  t_batch = t_train[batch_mask]

  x = np.arange(0.0, 20.0, 0.1)
  y = function_1(x)
  plt.xlabel("x")
  plt.ylabel("f(x)")
  plt.plot(x, y)
  plt.show()
