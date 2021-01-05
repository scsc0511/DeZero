import numpy as np

class Variable:
    def __init__(self, data):
        ## data와 grad는 모두 Numpy의 ndarray
        self.data = data
        self.grad = None    ## Backpropagation시 값이 정해지는 미분 값 

class Function:

    def __call__(self, inp):
        x = inp.data
        y = self.forward(x)
        output = Variable(y)    ## Variable Instance를 리턴
                                ## -> Function Instance의 연쇄적 호출 가능
                                ## -> 합성 함수 구현 가능 
        self.input = inp        ## Gradient를 구하기 위해 입력 변수 기억
        return output

    ## Function의 Sub Class에서 구현 
    ## Forward Propagation 수행(Predict 값 구하기)
    def foward(self, x):
        raise NotImplementedError()

    ## Function의 Sub Class에서 구현 
    ## Backward Propagation 수행(Gradient 값 구하기)
    def backward(self, x):
        raise NotImplementedError()


##Function Class의 Sub Class #1
class Square(Function):
    ## x = function의 input
    def forward(self, x):
        y = x**2
        return y

    ## gy = Chain Rule을 기반으로 구한 이전 부분 까지의 Gradient
    ## -> gy에 현재 Function의 미분 값을 곱해주면 구하고자하는 현
    ##    재 Gradient를 구할 수 있음 
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


##Function Class의 Sub Class #2
class Exp(Function):
    ## x = function의 input
    def forward(self, x):
        y = np.exp(x)
        return y

    ## gy = Chain Rule을 기반으로 구한 이전 부분 까지의 Gradient 
    ## -> gy에 현재 Function의 미분 값을 곱해주면 구하고자하는 현
    ## 재 Gradient를 구할 수 있음 
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

## Function Class 생성 (호출 아님)
A = Square()
B = Exp()
C = Square()

## Variable 생성 
x = Variable(np.array(0.5))

##Forward Propagation 수행 (호출 임)
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print(x.grad)
