import numpy as np

def as_array(x):
    if np.is_scalar(x):
        return np.array(x)
    return x


class Variable:
    def __init__(self, data):
        if (data is not None)\
        and not isinstance(data ,np.ndarray):
            raise TypeError('{}은(는) 지원하지 않습니다.'.format(type(data)))
  
        self.data = data 
        self.grad = None 
        self.creator = None 

    def set_creator(self, creator):
        self.creator = creator 

    def backward(self):
        if self.grad == None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while len(funcs) :
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function : 
    def __call__(self, inp):
       x = inp.data
       y = self.forward(x)

       outp = Variable(as_array(y))
       outp.set_creator(self)

       self.input = inp
       self.output = outp

       return outp

    def forward(self, x):
        raise NotImplementedError
 
    def backward(self, gy):
        raise NotImplementedError 

class Square(Function):
    def forward(self, x):
        return x**2
    
    def backward(self, gy):
        x = self.input.data
        gx = 2*x*gy

        return gx
         
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
 
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x)*gy
        return gx   

def square(x):
    f = Square()
    return f(x)

def exp(x):
    f = Exp()
    return f(x)

x = Variable(np.array(0.5))
a = square(x)
b = exp(a)
y = square(b)

#y.grad = np.array(1.0)
y.backward()
print(x.grad)


x = Varaible(0.5)
