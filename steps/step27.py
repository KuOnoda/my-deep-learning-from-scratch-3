if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
from dezero import Function
from dezero.utils import plot_dot_graph

import math

class Sin(Function):
    def forward(self,x):
        return np.sin(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.cos(x) * gy
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(10000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


x = Variable(np.array(np.pi / 4))
y = my_sin(x , threshold=1e-150) #thresholdによってグラフが変わる
y.backward()
print('--- approximate sin ---')
print(y.data)
print(x.grad)

x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='my_sin.png')