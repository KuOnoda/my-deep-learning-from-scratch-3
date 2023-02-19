import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

class Function: 
    def __call__(self,input):#f = Function() とした時、f(...)で__call__を呼び出せる
        x = input.data
        #y = x ** 2 #計算の内容例 
        y = self.forward(x) #具体的な計算
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()#Fuctionクラスは継承して利用すべきだと伝えるエラーを吐く

class Square(Function):
    def forward(self,x):
        return x ** 2

x = Variable(np.array(10))
#f = Function()
f = Square()
y = f(x)#__call__を呼ぶinput=x
print(type(y))
print(y.data)