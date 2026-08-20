from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    return np.exp(z_shifted) / np.sum(np.exp(z_shifted), axis=1, keepdims=True)

def softmax_single(z):
    return np.exp(z)/np.sum(np.exp(z))

def cross_entropy(y_true, y_pred):
    values = np.sum(y_true * y_pred, axis=1)
    return -np.mean(np.log(values+1e-15))

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
y_testing_onehot = np.zeros((x_testing.shape[0], 10))
for i in range(x_testing.shape[0]):
    y_testing_onehot[i][y[i+60000]] = 1

W = np.load('W.npy')
B = np.load('B.npy')

scores = x_training @ W + B
scores = softmax(scores)
c = cross_entropy(y_onehot, scores)

print('Cross-entropy/cost: ' + str(c))

correct = 0
conf = 0
total = 0

for k in range(x_testing.shape[0]):
    predicted = softmax_single(x_testing[k] @ W + B)
    digit = np.argmax(predicted)
    confidence = predicted[digit]
    check = np.argmax(y_testing_onehot[k])
    correct += (digit == check)
    total += 1
    conf += confidence

print('Accuracy: ' + str(100*correct/total)+'%')
print('Confidence: ' + str(100*conf/total)+'%')