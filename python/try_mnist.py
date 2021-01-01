#%%
import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append('./deep-learning-from-scratch')
from dataset.mnist import load_mnist

#%%
def img_show(img):
  plt.imshow(img)

#%%
(x_train, t_train), (x_test, t_test) = \
  load_mnist(flatten=True, normalize=False)

#%%
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

#%%
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
