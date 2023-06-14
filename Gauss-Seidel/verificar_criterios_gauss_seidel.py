# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:09:13 2023

@author: alanm
"""

 
import Sistema as sis 
import numpy as np
import emoji

def is_gauss_seidel_convergente(A):
    diagonal = np.abs(np.diag(A))
    triangular_inferior = np.tril(A, k=-1)
    return np.all(diagonal > np.sum(np.abs(triangular_inferior), axis=1))

# Exemplo de uso
A = sis.A

is_convergente = is_gauss_seidel_convergente(A)
print(A)

if is_convergente == True:
    print(emoji.emojize("O sistema satisfaz o critério de convergência do Método de Gauss-Seidel? :check_mark:"))
else :
    print(emoji.emojize("O sistema satisfaz o critério de convergência do Método de Gauss-Seidel? :cross_mark:"))
    