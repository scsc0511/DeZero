from step1_variable import Variable
from step2_function import Function, Square
import numpy as np

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

#Function Class의 __call__ 메서드는 input과 output이 모두 
#Variable Object이므로 여러 Function을 연이어 사용할 수 있음 
#Class.__init__()가 호출됨 
##A = Square()
##B = Exp()
##C = Square()

#Class.__call__(Variable)이 호출됨 
##x = Variable(np.array(0.5))
##a = A(x)
##b = B(a)
##y = C(b)
##print(y.data)


#일련의 계산을 Computation Graph로 표현하는 이유 
#Computation Graph를 사용하면 각 변수에 대한 미분을 효율적으로 계산할 수 있기 때문 
