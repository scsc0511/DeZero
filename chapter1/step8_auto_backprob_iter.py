import numpy as np

class Variable :

    def __init__(self, data):
        self.data = data 
        self.grad = None 
        self.creator = None 

    def set_creator(self, creator):
        self.creator = creator

    def backward(self):
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)

class Function :
    def __call__(self, inp):
        x = inp.data
        y = self.forward(x)

        outp = Variable(y)
        outp.set_creator(self)    

        self.input = inp
        self.output = outp

        return outp

    def forward(self, x):
        raise NotImplementedError

    def bakcward(self, gy):
        raise NotImplementedError

class Square(Function) :

    def forward(self, x):
        y = x**2
        return y
 
    def backward(self, gy):
       x = self.input.data
       gx = 2*x*gy 

       return gx 

class Exp(Function):
  
    def forward(self, x):
        y = np.exp(x)
        return y

    def backward(self, gy):
        x = self.input.data 
        gx = np.exp(x)*gy
   
        return gx

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(x.grad)
