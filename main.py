import numpy as np
import matplotlib.pyplot as plt
print('hello world')
np.random.seed(42)
m_i = 3
b_i = 5
x_list = np.random.rand(100) * 10
y_list = m_i * x_list + b_i + np.random.normal(0, 2, 100)
learning_rate = 0.001
steps = 10000

m = 0
b = 0

def cost_calc(x, y, m, b):
    total = 0
    for i in range(len(x)):
        total += (y[i] - (m * x[i] + b)) ** 2
    return total / len(x)

print(cost_calc(x_list, y_list, m, b))
for k in range(steps):
    m_gradient = 2/100 * sum(x_list[i] * (y_list[i] - (m * x_list[i] + b)) for i in range(len(x_list)))
    b_gradient = 2/100 * sum(y_list[i] - (m * x_list[i] + b) for i in range(len(x_list)))
    m += learning_rate * m_gradient
    b += learning_rate * b_gradient

print(cost_calc(x_list, y_list, m, b))
plt.scatter(x_list, y_list)
y_pred = m * x_list + b
print(m,b)
plt.plot(x_list, y_pred, color='green')
plt.show()