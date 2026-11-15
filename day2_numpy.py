import numpy as np
a = np.array([1, 2, 3, 4 , 5])
print(a * 2)
print(a + 10)

m = np.array([[1,2,3],[4,5,6]])

print(m.shape)
print(m.T)
print(m.sum(axis=0))
print(m.mean())

print(np.zeros((2,3)))
print(np.arange(0, 10, 2))

print(a[a > 2])
