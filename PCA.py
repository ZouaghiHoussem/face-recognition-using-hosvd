# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

P = 3 #Broj osoba
I = 5 #broj slika
imgShape = scipy.misc.imread('att_faces/orl_faces/s1/1.pgm').shape
imgVecSize = scipy.misc.imread('att_faces/orl_faces/s1/1.pgm').ravel().size


def LoadData(P,I,imgSize):
    T = np.zeros((P, imgSize, I))
    for i in range(1,P+1):
        for j in range(1,I+1):
            tmp = scipy.misc.imread('att_faces/orl_faces/s'+str(i)+'/'+str(j)+'.pgm')
            T[i-1,:,j-1] = tmp.ravel()
    return T


def main():
    T = LoadData(P,I,imgVecSize)
    #print(T.shape)
    U, S, V = np.linalg.svd(T, full_matrices=False)
    #print(U.shape, S.shape, V.shape)
    D = LoadData(P+2,10,imgVecSize)
    
    
    for i in range(P):
        for j in range(10):
            tmp = np.dot(U[i,:,:], np.dot(U[i,:,:].T, D[i,:,j].ravel())) - D[i,:,j].ravel()
            print(str(j+1)+":",np.linalg.norm(tmp))
        print('\n')

    
if __name__ == "__main__":
    main()