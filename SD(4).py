from sympy import *
from sympy.solvers import solve
import numpy as np
from colorama import init, Fore

init(autoreset=True)

def grad_k(grad, k):
    gr = []
    for i in grad:
        gr.append(i(k))
    return gr

def sub_list(l1, l2):
	list_sub = []
	for i in range(len(l1)):
		list_sub.append(l1[i]-l2[i])
	return list_sub

def mat_list(alist, x):
	return [x*i for i in alist]

def optimal_solution(xk, f, grad, t):
	numerical_grad = grad_k(grad, xk)
	t_grad = mat_list(numerical_grad, t)
	phi_t = f(sub_list(xk, t_grad))
	roots = solve(phi_t.diff(t))
	phi = lambdify(t, phi_t, 'numpy')
	tk = roots[0]
	for i in roots:
		if(phi(i)<phi(tk)):
			tk = i
	return sub_list(xk, mat_list(numerical_grad, tk))

def SD_iteration(x0, f, grad, t, max_iter):
	print(Fore.GREEN + "Steepest Descent by iteration:")
	x_round = x0
	for i in range(max_iter):
		x_round = optimal_solution(x_round, f, grad, t)
		print(f'	Step {i+1} : {x_round}')
	return x_round

def SD_epsilon(x0, f, grad, t, epsilon):
	print(Fore.YELLOW + "Steepest Descent by ε:")
	x_round = x0
	i = 1
	while True:
		prev_x = x_round
		x_round = optimal_solution(x_round, f, grad, t)
		print(f'	Step {i} : {x_round}')
		i+=1
		if(abs(f(prev_x)-f(x_round))<epsilon):
			return x_round

#User Inputs
n_var = int(input("enter the number of variables: "))
x0 = [float(i) for i in input("enter x0 : ").split(",")]
f = sympify(input("enter the expression: "))
max_iter = int(input("enter maximum iterations required: "))
epsilon = float(sympify(input("enter ε: ")))

t = Symbol("t")
var = [Symbol(f'x{i+1}') for i in range(n_var)]
grad = [f.diff(var[i]) for i in range(n_var)]
grad = [lambdify([var], i, 'numpy') for i in grad]
f = lambdify([var], f, 'numpy')

SD_epsilon(x0, f, grad, t, epsilon)
SD_iteration(x0, f, grad, t, max_iter)