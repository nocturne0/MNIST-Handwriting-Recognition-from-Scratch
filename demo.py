from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
import os

warnings.filterwarnings('ignore', category=RuntimeWarning)

def softmax_single(z):
    return np.exp(z)/np.sum(np.exp(z))

def display_image(image_array):
    chars = " .:-=+*#%@"
    output = ""
    for row in range(28):
        line = ""
        for col in range(28):
            pixel_value = image_array[row][col]
            line += chars[int(pixel_value * (len(chars) - 1))]
        output += line
        if(row < 27):
            output += "\n"
    return output

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

correct = 0
total = 0

os.system('cls' if os.name == 'nt' else 'clear')
input('')
os.system('cls' if os.name == 'nt' else 'clear')

while True:
    random_index = np.random.randint(0, x_testing.shape[0])
    predicted = softmax_single(x_testing[random_index] @ W + B)
    digit = np.argmax(predicted)
    confidence = predicted[digit]
    check = np.argmax(y_testing_onehot[random_index])
    correct += (digit == check)
    total += 1
    image_array = x_testing[random_index].reshape(28, 28)
    print(display_image(image_array))
    print('Predicted: ' + str(digit))
    print('Confidence: ' + str(round(100*confidence, 2))+'%')
    print('Actual: ' + str(check))
    print('Accuracy: ' + str(correct) +'/' + str(total) + ' (' + str(round(100*correct/total, 2))+'%)')
    z = input('')
    if z == 'stop':
        break
    os.system('cls' if os.name == 'nt' else 'clear')