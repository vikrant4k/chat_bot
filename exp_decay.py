import math
lamb=1e-4
N=0.99
for i in range(10000):
    n_t=math.exp(-lamb*i)
    print(i,n_t)
