import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None #変数を生み出した関数とのつながり

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator#関数取得
        if f is not None:
            x = f.input#関数の入力を取得
            x.grad = f.backward(self.grad)#関数のbackwardメソッド
            x.backward() #（再帰）自分よりひとつ前のbackwardを呼ぶ

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)#生みの親として自分を設定
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()
    


class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

#assert y.creator == B #True出ない場合はエラーを吐く

#順番に処理する場合
y.grad = np.array(1.0)
C = y.creator
b = C.input
b.grad = C.backward(y.grad)

B = b.creator
a = B.input
a.grad = B.backward(b.grad)

A = a.creator
x = A.input
x.grad = A.backward(a.grad)
print(x.grad)

#backwardメソッドを使用
y.grad = np.array(1.0)
y.backward()
print(x.grad)