import TheanoCode
from TheanoCode import tensor

a = tensor.dscalar()
b = tensor.dscalar()

c = a + b

f = TheanoCode.function([a, b], c)

result = f(10, 20)

print(result)