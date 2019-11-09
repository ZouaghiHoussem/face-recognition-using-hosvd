#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:44:41 2017

@author: marin
"""
#import tensorflow as tf
#import scipy as sc

import numpy as np
import tensorly as tly
from tensorly.tenalg import mode_dot
tly.set_backend('numpy')

#   tly.set_backend('mxnet')
#   tly.set_backend('pytorch')


def HOSVD(A):
    #HOSVD tenzora 3 ranka
    A = tly.tensor(A)
    
    
    #mora biti dan argument full_matrices inace se funkcija za svd srusi
    U1, s1, v = np.linalg.svd(tly.unfold(A,0), full_matrices=False)
    U2, s2, v = np.linalg.svd(tly.unfold(A,1), full_matrices=False)
    U3, s3, v = np.linalg.svd(tly.unfold(A,2), full_matrices=False)
    
    """
    U1, s1, v = np.linalg.svd(tly.unfold(A,0), full_matrices=True)
    U2, s2, v = np.linalg.svd(tly.unfold(A,1), full_matrices=True)
    U3, s3, v = np.linalg.svd(tly.unfold(A,2), full_matrices=True)
    """
    
    #S = np.tensordot(np.tensordot(np.tensordot(A,U1.T,0), U2.T,1), U3.T,2)
    S = mode_dot(mode_dot(mode_dot(A,U1.T,0), U2.T,1), U3.T,2)
    
    return S, U1, U2, U3