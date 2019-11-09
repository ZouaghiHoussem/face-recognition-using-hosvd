#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 22:13:55 2017

@author: marin
"""

import numpy as np
#from unfoldModules import __get_unfolding_mode_order

def __get_unfolding_mode_order(A, n):
    return [i for i in range(n+1, A.ndim)] + [i for i in range(n)]

def __get_unfolding_stride(A, mode_order):
    stride = [0 for i in range(A.ndim)]
    stride[mode_order[A.ndim-2]] = 1
    for i in range(A.ndim-3, -1, -1):
        stride[mode_order[i]] = (
            A.shape[mode_order[i+1]] * stride[mode_order[i+1]])
    return stride

def __get_tensor_indices(r, c, A, n, mode_order, stride):
    i = [0 for j in range(A.ndim)]
    i[n] = r
    i[mode_order[0]] = c / stride[mode_order[0]]
    for k in range(1, A.ndim-1):
        i[mode_order[k]] = (
            (c % stride[mode_order[k-1]]) / stride[mode_order[k]])
    return i

def get_unfolding_matrix_size(A, n):
    row_count = A.shape[n]    
    col_count = 1   
    for i in range(A.ndim):
        if i != n: col_count *= A.shape[i]        
    return (row_count, col_count)

def unfold(A, n):
    """
    Unfold tensor A along Mode n
    """
    (row_count, col_count) = get_unfolding_matrix_size(A, n)
    result = np.zeros((row_count, col_count))
     
    mode_order = __get_unfolding_mode_order(A, n)
    stride = __get_unfolding_stride(A, mode_order)
         
    for r in range(row_count):        
        for c in range(col_count):
            i = __get_tensor_indices(r, c, A, n, mode_order, stride)
            print(i)
            result[r, c] = A.__getitem__(tuple(i))
    return result