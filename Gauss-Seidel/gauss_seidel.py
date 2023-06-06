# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 15:49:32 2023

@author: alanm
"""
import Sistema as sys
import numpy as np
import matplotlib.pyplot as plt

# Função do método de Gauss-Seidel
def gauss_seidel(A, b, x0, max_iter=100, error=0.001):
    n = len(b)
    x = x0.copy()
    errors = []
    for _ in range(max_iter):
        x_new = np.zeros_like(x)
        for i in range(n):
            s1 = np.dot(A[i, :i], x_new[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
        errors.append(np.linalg.norm(x - x_new))
        if errors[-1] < error:
            break
        x = x_new.copy()
    return x, errors

# Chutes iniciais
x0_valores = np.random.rand(10, len(sys.b))

# Soluções e erros usando o método de Gauss-Seidel
solucoes_seidel = []
erros_seidel = []

for x0 in x0_valores:
    solucao, erros = gauss_seidel(sys.A, sys.b, x0)
    solucoes_seidel.append(solucao)
    erros_seidel.append(erros)

#exibir chutes 
print(x0_valores)

# Plotando os gráficos de iterações para o método de Gauss-Seidel
plt.figure(figsize=(10, 6))
for i in range(len(x0_valores)):
    plt.plot(range(len(erros_seidel[i])), erros_seidel[i], label=f'Solução {i+1}: {solucoes_seidel[i]}')
plt.xlabel('Iteração')
plt.ylabel('Erro')
plt.title('Gráfico de Iterações - Gauss-Seidel')
plt.legend()
plt.grid(True)
plt.show()

