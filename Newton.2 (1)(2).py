from sympy import *
import numpy as np
import matplotlib.pyplot as plt

def Hessian(grad, var, n_var):
    Hess = []
    for i in range(n_var):
        for j in range(n_var):
            x = grad[i].diff(var[j])
            Hess.append(lambdify([var], x, 'numpy'))
    return Hess

def grad_k(grad, k):
    gr = []
    for i in grad:
        gr.append(i(k))
    return gr

def Hess_k(Hess, k):
    Hs = []
    for i in Hess:
        Hs.append(i(k))
    return Hs

def matrix(alist, n, m):
    x = np.asarray(alist)
    x = x.reshape((n,m))
    x = np.asmatrix(x)
    return x

def newton(n_var, var, grad, Hess, f, epsilon, max_iter, xk):
    
    for i in range(max_iter):
        xk_prev = xk
        Hess_inv = np.linalg.inv(matrix(Hess_k(Hess, xk), n_var, n_var))
        grad_matrix = matrix(grad_k(grad, xk), n_var, 1)
        xk = matrix(xk, n_var, 1) - np.matmul(Hess_inv, grad_matrix)
        xk = [i[0] for i in xk.tolist()]

        # if(abs(f(xk)-f(xk_prev))<epsilon):
        #     return xk
    return xk

#User inputs
n_var = int(input("enter the number of variables: "))
xk = [float(i) for i in input("enter x0 : ").split(",")]
f = sympify(input("enter the expression: "))
max_iter = int(input("enter maximum iterations required: "))
epsilon = sympify(input("enter ε: "))

var = [Symbol(f'x{i+1}') for i in range(n_var)]
grad = [f.diff(var[i]) for i in range(n_var)]
Hess = Hessian(grad, var, n_var)
grad = [lambdify([var], i, 'numpy') for i in grad] 
f = lambdify([var], f, 'numpy')

print(newton(n_var, var, grad, Hess, f, epsilon, max_iter, xk))