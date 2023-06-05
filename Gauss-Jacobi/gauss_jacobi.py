# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:12:15 2023

@author: alanm
"""

import numpy as np
import Sistema as sis
import matplotlib.pyplot as plt

# Função do método de Gauss-Jacobi
def gauss_jacobi(A, b, x0, max_iter=10, tolerancia=0.001):
    n = len(b)
    x = x0.copy()
    erros = []
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s = np.dot(A[i, :i], x[:i]) + np.dot(A[i, i+1:], x[i+1:])
            x_new[i] = (b[i] - s) / A[i, i]
        erros.append(np.linalg.norm(x - x_new))
        if erros[-1] < tolerancia:
            break
        x = x_new.copy()
    return x, erros


#*************************************************************

# Questão B
# Escolha de 10 diferentes valores de x0
x0_valores = np.random.rand(10, len(sis.b))

# Soluções e erros usando o método de Gauss-Jacobi
solucoes = []
erros = []

for x0 in x0_valores:
    solucao, erro = gauss_jacobi(sis.A, sis.b, x0)
    solucoes.append(solucao)
    erros.append(erro)

# Exibindo as soluções
for i, solucao in enumerate(solucoes):
    print(f"Solução {i+1}: {solucao}")

# Plotando os gráficos de iterações para o método de Gauss-Jacobi
plt.figure(figsize=(10, 10))
for i in range(len(x0_valores)):
    plt.plot(range(len(erros[i])), erros[i], label=f'X0={x0_valores[i]}')
plt.xlabel('Iteração')
plt.ylabel('Erro')
plt.title('Gráfico de Iterações - Gauss-Jacobi')
plt.legend()
plt.grid(True)
plt.show()

#fim