from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

mnist = fetch_openml('mnist_784', version=1)
x = mnist.data.to_numpy()
y = mnist.target.to_numpy()
y = y.astype(int)
x = x / 255.0

x_training = x[:60000]
x_testing = x[60000:]

y_onehot = np.zeros((x_training.shape[0], 10))
for i in range(x_training.shape[0]):
    y_onehot[i][y[i]] = 1

print(x.shape)
print(y.shape)

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    return np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=1, keepdims=True)

def cross_entropy(y_true, y_pred):
    values = np.sum(y_true * y_pred, axis=1)
    return -np.mean(np.log(values+1e-15))

def gradient(x, y_true, y_pred):
    xT = x.transpose()
    n = x.shape[0]
    grad_W = 1/n * xT @ (y_pred - y_true)
    grad_B = 1/n * np.sum(y_pred - y_true, axis = 0)
    return grad_W, grad_B
    
# W = np.zeros((784, 10))
# B = np.zeros((10))
W = np.load('W.npy')
B = np.load('B.npy')

scores = x_training @ W + B
scores = softmax(scores)
c = cross_entropy(y_onehot, scores)

print(c)

learning_rate = 0.01
learning_steps = 1000

for k in range(learning_steps):
    if(k % 100 == 0):
        print('\r' +str(k)+' steps (' + str(100 * k / learning_steps) + '%) done, loss: ' + str(c), end='')
    g_W, g_B = gradient(x_training, y_onehot, scores)
    W -= learning_rate * g_W
    B -= learning_rate * g_B
    scores = x_training @ W + B
    scores = softmax(scores)
    c = cross_entropy(y_onehot, scores)
np.save('W.npy', W)
np.save('B.npy', B)
print('\nTraining Complete')
print(c)