import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

class Function: 
    def __call__(self,input):#f = Function() とした時、f(...)で__call__を呼び出せる
        x = input.data
        y = self.forward(x) #具体的な計算
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()#Fuctionクラスは継承して利用すべきだと伝えるエラーを吐く

class Square(Function):
    def forward(self,x):
        return x ** 2

class Exp(Function):
    def forward(self,x):
        return np.exp(x)

def numerical_diff(f,x,eps=1e-04): #中心差分近似
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2*eps)

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f,x)

print(dy)

"""合成関数の微分"""
def f(x):
    A = Square()
    B = Exp()
    C = Square()
    return C(B(A(x)))


x = Variable(np.array(0.5))
dy = numerical_diff(f, x)
print(dy)