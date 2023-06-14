# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 14:36:08 2023

@author: alanm
"""

import numpy as np
import Sistema as sis
import emoji

def is_gauss_jacobi_convergente(A):
    diagonal = np.abs(np.diag(A))
    outros_elementos = np.sum(np.abs(A), axis=1) - diagonal
    return np.all(diagonal > outros_elementos)

# Exemplo de uso
A = sis.A

is_convergente = is_gauss_jacobi_convergente(A)
print(A)

if is_convergente == True:
    print(emoji.emojize("O sistema satisfaz o critério de convergência do Método de Gauss-Jacobi? :check_mark:"))
else :
    print(emoji.emojize("O sistema satisfaz o critério de convergência do Método de Gauss-Jacobi? :cross_mark:"))
    