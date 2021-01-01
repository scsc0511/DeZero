#1. 기본 내용 

## Numerical Differentiation
### 컴퓨터는 극한을 취급할 수 없으니 h를 무한소에 가까운 아주 작은 값을 
### 이용하여 미분값 즉 함수의 변화량을 구하는 방법  

## Centered Difference
### Numerical Differentiation에서 무한소와 h로 지정된 아주 작은 값 사이의 
### 차이로 인해 발생하는 오차를 줄이기 위해 사용될 수 있는 방법
### Forward Difference에서는 f(x+h)-f(x)를 분자로 하지만 
### Centered Difference에서는f(x+h) - f(x-h)를 분자로 함. 
### 그리고 이에따라 분모가 2h가 됨   

from step1_variable import Variable
from step2_function import Function 


## Centered Difference를 이용하여 Numerical Differentiation을 하는 함수
### f = 미분의 대상이 되는 함수, Function Instance 
### x = f를 미분하는 변수, Variable Instance 
### eps = h에 대응하여 무한소를 대체하는 작은값, 디폴트로 1e-4
def numerical_diff(f, x, eps=1e-04):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2*eps) 


#2. Example
from step2_function import Square
import numpy as np

f = Square()
x = Variable(np.array(2.0))
dy = numerical_diff(f, x)
print(dy)

from step3_connect_function import Exp

#3. 합성 함수의 미분 
def f(x):
    A = Square()
    B = Exp()
    C = Square()

    return C(B(A(x)))

x = Variable(np.array(0.5))
dy = numerical_diff(f,x)
print(dy)


#4. Numerical Differentiation의 문제점 
## Numerical Differentiation에서는 h를 무한소가 아닌 아주 작은 값(ex : 
## 0.0001)을 사용하기 때문에 오차가 포함될 수 밖에 없음. 대부분의 경우 
## 오차가 작지만 때로는 커질 수도 있는데 이는 주로 자릿수 누락 때문임
## Ex : f(x+h) = 1.234 , f(x) = 1.233이라고 하면 Numerical Differentia
##      tion의 결과는 0.001/0.001 = 1임 but 실제 f(x+h) = 1.234......
##      이고 f(x) = 1.233........이어서 f(x+h)-f(x) = 0.001434.......
##      와 같은 결과였을지도 모름   
## 
## Numerical Differentiation에서는 계산량이 많음. 이는 변수가 여러개인 
## 계산을 미분할 경우 변수 각각을 미분해야 하기 때문임. 신경망에서는 
## 수백만개의 Parameter를 사용하는 일이 흔하므로 이를 모두 Numerical 
## Differntiation으로 구하는 것은 현실적이지 않음 
##
## cf: Numerical Difference의 활용 - Gradient Checking
## Numerrical Difference는 구현하기 쉽고 거의 정확한 값을 얻을 수 있는
## 반면 Back Propagation은 알고리즘이 복잡하여 버그가 발생하기 쉽기 때
## 문에 Back Propagation을 정확하게 구현했는지를 확인하기 위해 Numeric
## al Differentiation의 결과를 사용
