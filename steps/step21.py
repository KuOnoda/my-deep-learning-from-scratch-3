import numpy as np
import weakref
import contextlib


class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)#ゲッター 名前を取り出す(前のやつ保持)
    setattr(Config, name,value)#セッター nameの属性がvalueに設定される
    try:
        yield
    finally:
        setattr(Config, name,old_value)

def no_grad():
    return using_config('enable_backprop', False)


class Variable:
    def __init__(self, data, name=None):
        if data is not None: #ndarray以外の型にはエラー（Noneは許可）
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0 #世代の追加

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):#ループで実装
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
            #gys = [output.grad for output in f.outputs]
            gys = [output().grad for output in f.outputs] #output()で参照先の値を得る
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
                
                if not retain_grad:
                    for y in f.outputs:
                        y().grad = None #微分が入らない部分は保持しない

    def cleargrad(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype
    
    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' *9)
        return 'variable(' + p + ')'

    def __mul__(self,other):
        return mul(self,other)


class Function:
    def __call__(self, *inputs): #可変長引数
        inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]
        
        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs] #アドレスを渡す
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

class Mul(Function):
    def forward(Self,x0,x1):
        y = x0 * x1
        return y
    
    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0

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
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0,x1):
    x1 = as_array(x1)
    return Mul()(x0,x1)

def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

Variable.__add__ = add
Variable.__radd__ = add
Variable.__mul__ = mul
Variable.__rmul__ = mul

a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

# y = add(mul(a, b), c)
y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)