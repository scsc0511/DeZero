# Define by Run 
# 딥러닝에서 수행하는 계산 시점에 함수와 변수들을 연결하는 방식
import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data 
        self.grad = None
        self.creator = None 
   
    def set_creator(self, func):
        self.creator = func 

class Function:
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

    def backward(self, x):
        raise NotImplementedError

class Square(Function):
    def forward(self, x):
        y =x**2
        return y

    def backward(self, gy):
        x = self.input.data
        # current gradient * future gradient
        gx = 2 * x * gy
        return gx 

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
 
    def backward(self, gy):
        x = self.input.data
        # current gradient * future gradient 
        gx = np.exp(x) * gy  
        return gx
        
A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

