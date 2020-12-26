import numpy as np

a=np.array([[1,2],[1,1],[1,1],[2,1]])
b=np.array([1,1,2,1,2,1])


x1=np.array([1,2,3])
x2=np.array([4,5,6])

x3=np.array([[1,2],[3,4]])
x4=np.array([[3,4],[5,5]])

print(np.dot(x1,x2))
print(np.dot(x3,x4))
print(x3*x4)
print(np.mat(x3)*np.mat(x4))
print(np.multiply(x3,x4))
print(np.multiply(np.mat(x3),np.mat(x4)))

print(np.matmul(x3,x4))
print(np.matmul(np.mat(x3),np.mat(x4)))

print(np.matmul(x1,x2))
print(np.dot(x1,x2))