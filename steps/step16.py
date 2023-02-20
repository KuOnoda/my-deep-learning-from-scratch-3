import numpy as np


class Variable:
    def __init__(self, data):
        if data is not None: #ndarray以外の型にはエラー（Noneは許可）
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 #世代の追加

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self):#ループで実装
        if self.grad is None:#y.grad = np.array(1.0)を省略
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()#集合
        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x : x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            
            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator) 

    def cleargrad(self):
        self.grad = None


class Function:
    def __call__(self, *inputs): #可変長引数
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs  # Set output
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx

class Add(Function):
    def forward(self,x0,x1):
        y = x0 + x1
        return y
    
    def backward(self, gy): #足し算の逆伝播は分岐してそのまま流れる
        return gy,gy

#Pythonの関数の形で使えるようにする
def square(x):
    return Square()(x)

def exp(x):
    return Exp()(x)

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

def add(x0,x1):
    return Add()(x0, x1)

x = Variable(np.array(2.0))
a = square(x)
y = add(square(a),square(a))
y.backward()

print(y.data)
print(x.grad)