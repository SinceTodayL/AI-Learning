import pandas
import numpy as np


# 创建数组
a = np.array([2, 3, 4], dtype=float)
b = np.array([[2, 3], [3, 4], [4, 5]])

c = np.linspace(1, 10, 2)

print(a)
print(b)
print(c)

a_random = np.random.rand(2, 3)   # (2, 3) is shape, same as torch
b_random = np.random.randn(2, 3)  # normal distri.
c_random = np.random.randint(4, 7, (2, 3)) # random int
d_random = np.random.permutation(10) 

print(a_random, b_random, c_random, d_random)

f = [1, 10, 100, 1000]
p = [0.1, 0.2, 0.3, 0.4]
record = []
for i in range(1000000):
    record.append(np.random.choice(f, p=p))
print(record.count(1), record.count(10), record.count(100), record.count(1000))

