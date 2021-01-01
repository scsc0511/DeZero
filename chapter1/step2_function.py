from step1_variable import Variable
import numpy as np

#Definition of Class Function
#Funcion Class는 Super Class로서 모든 Function이 공통으로 지니는 기능 구현
#각 Fucntion의 구체적인 기능은 Function Class를 상속받은 Sub Class에서 구현
class Function:
    #Variable에서 데이터 찾음
    #계산 결과를 Variable에 포장 
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x**2

#Excercise of Class Function 
##x = Variable(np.array(10))
##f = Function()
##y = f(x)

#print(type(y))
#print(y.data)

##f = Square()
##y = f(x)
##print(type(y))
##print(y.data)

