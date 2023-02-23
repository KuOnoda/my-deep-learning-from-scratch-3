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


#合成関数
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
c = C(b)

y = C(b)
print(y.data)
y = C(B(A(x)))
print(y.data)