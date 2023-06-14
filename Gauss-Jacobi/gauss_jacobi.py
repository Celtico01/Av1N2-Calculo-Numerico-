# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 16:12:15 2023

@author: alanm
"""

import numpy as np
import sys
import Sistema as sysL
import matplotlib.pyplot as plt
import verificar_criterios_gauss_jacobi as verificar

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

if verificar.is_convergente == False:
    
    print('O sistema não cumpre os critérios de Gauss-Jacobi.')
    sys.exit()

# Questão B
# Escolha de 10 diferentes valores de x0
np.random.seed(42)
x0_valores = np.random.rand(10, len(sysL.b))

# Soluções e erros usando o método de Gauss-Jacobi
solucoes = []
erros = []

for x0 in x0_valores:
    solucao, erro = gauss_jacobi(sysL.A, sysL.b, x0)
    solucoes.append(solucao)
    erros.append(erro)

# Exibindo os x0
print("Valores de X0:")
print(x0_valores)

# Plotando os gráficos de iterações para o método de Gauss-Jacobi
plt.figure(figsize=(10, 10))
for i in range(len(x0_valores)):
    plt.plot(range(len(erros[i])), erros[i], label=f'Solução {i+1}: {solucoes[i]}')
plt.xlabel('Iteração')
plt.ylabel('Erro')
plt.title('Gráfico de Iterações - Gauss-Jacobi')
plt.legend()
plt.grid(True)
plt.show()

#fim