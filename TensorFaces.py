# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
#import scipy as sc
from random import randint
#import imageio
#import tensorflow as tf
from HOSVD import HOSVD
import tensorly as tly
import time
tly.set_backend('numpy')
#   tly.set_backend('mxnet')
#   tly.set_backend('pytorch')

#from tensorly.decomposition import tucker
#   from tensorly import tucker_to_tensor
from tensorly.tenalg import mode_dot

#from tensorly.backend.SVDFullMatricesFlag import setSVDFlag
#setSVDFlag(False)

P = 40   #Broj osoba
I = 7   #broj slika
tol = 0.9
imgShape = img.imread('att_faces/orl_faces/s1/1.pgm').shape
imgVecSize = img.imread('att_faces/orl_faces/s1/1.pgm').ravel().size


def LoadData(P,I,imgSize):
    #ucitaj osobe u tenzor
    T = np.zeros((P, imgSize, I))
    D = np.zeros((P, imgSize, 10-I))
    for i in range(1,P+1):
        for j in range(1,I+1):
            tmp = img.imread('att_faces/orl_faces/s'+str(i)+'/'+str(j)+'.pgm')
            #T[i-1,:,j-1] = np.fft.fft(tmp.ravel()) #fourier
            T[i-1,:,j-1] = tmp.ravel()
        for j in range(I+1,11):
            tmp = img.imread('att_faces/orl_faces/s'+str(i)+'/'+str(j)+'.pgm')
            D[i-1,:,j-I-1] = tmp.ravel()
    return T, D

def QR(B):
    #QR dekompozicija
    Qs = []
    Rs = []
    for i in range(B.shape[2]):
        Qe, Re = np.linalg.qr(B[:,:,i])
        #Qe, Re = sc.linalg.qr(B[:,:,i])
        Qs.append(Qe)
        Rs.append(Re)
    return Qs, Rs

"""
def main():
    # Dobar primjer1
    #Pers = 9
    #Expr = 6
    # Dobar primjer2
    #Pers = 5
    #Expr = 9
    Pers = 8
    Expr = 3
    
    def PlotImg(p, e):
        ReconstructedImg1 = np.dot(np.dot(F, mode_dot(S,H[:,p],0)), G[:,e])
        ReconstructedImg2 = np.dot(np.dot(F, mode_dot(S,H[p,:],0)), G[e,:])
        TestImg = D[Pers,:,Expr]
        PredictImg = T[p,:,e]
        plt.figure()
        plt.imshow(TestImg.reshape(imgShape))
        plt.figure()
        plt.imshow(PredictImg.reshape(imgShape))
        plt.figure()
        plt.imshow(ReconstructedImg1.reshape(imgShape))
        plt.figure()
        plt.imshow(ReconstructedImg2.reshape(imgShape))
        
        
    T = LoadData(P,I,imgVecSize)
    D = LoadData(P+2,10,imgVecSize)
        
    S, U_list = tucker(T)
    H = U_list[0]; F = U_list[1]; G = U_list[2]
    #S, H, F, G = HOSVD(T)
        
    B = mode_dot(S,G,2)
    
    def TestImg(z):
        predict = []
        norms = []
        Predict_pers = -2
        Predict_expr = -2
        Z = np.dot(F.T, z)
        for k in range(B.shape[2]):
            alfa = np.linalg.lstsq(B[:,:,k].T, Z)
            #alfa = np.linalg.lstsq(Qs[0], np.dot(Rs[0],Z))
            for i in range(H.shape[0]):
                norm = np.linalg.norm(alfa[0]-H[i,:])
                #print(str(i+1)+':  '+str(norm))
                if norm < tol:
                    predict.append(i)
                    norms.append(norm)
            #print('\n')
        #print(predict); print(norms)
        if len(predict) == 0:
            Predict_pers = -1
            Predict_expr = -1
        else:
            Predict_pers = np.mean(predict)
            Predict_expr = norms.index(min(norms))
            #print(Predict_pers, Predict_expr); print(min(norms))
        if Predict_pers == -1:
            print('Osoba nije u bazi podataka')
        return Predict_pers, Predict_expr
    
    def Accuracy():
        accNum = 0
        falsNum = 0
        for p in range(D.shape[0]-2):
            for e in range(D.shape[2]):
                z = D[p,:,e]
                pp, pe = TestImg(z)
                print(pp, p)
                if pp-p == 0:
                    accNum = accNum+1
                else:
                    falsNum = falsNum+1
        return accNum/(accNum+falsNum)
     
    print(Accuracy())
    
    #z = D[Pers,:,Expr]
    #Predict_pers, Predict_expr = TestImg(z)
    #PlotImg(int(Predict_pers), Predict_expr) 


if __name__ == "__main__":
    main()
"""



#------------MAIN------------------
    
T, D = LoadData(P,I,imgVecSize) #baza podataka
   
start_train = time.time() 
#S, U_list = tucker(T)
#H = U_list[0]; F = U_list[1]; G = U_list[2]
S, H, F, G = HOSVD(T)
end_train = time.time()
    
B = mode_dot(S,G,2)
Qs, Rs = QR(B)

def TestImg(z):
    predict = []
    norms = []
    Predict_pers = -2
    Predict_expr = -2
    Z = np.dot(F.T, z)
    for k in range(B.shape[2]):
        alfa = np.linalg.lstsq(B[:,:,k].T, Z)
        #alfa = np.linalg.lstsq(Rs[k], np.dot(Qs[k].T,Z))
        for i in range(H.shape[0]):
            norm = np.linalg.norm(alfa[0]-H[i,:])
            #print(str(i+1)+':  '+str(norm))
            if norm < tol:
                predict.append(i)
                norms.append(norm)
        #print('\n')
    #print(predict); #print(norms)
    if len(predict) == 0:
        Predict_pers = -1 #-1 ako osoba nije u bazi
        Predict_expr = -1
    else:
        Predict_expr = norms.index(min(norms))
        Predict_pers = predict[Predict_expr]
        #print(Predict_pers, Predict_expr); print(min(norms))
    if Predict_pers == -1:
        print('Osoba nije u bazi podataka')
    return Predict_pers, Predict_expr


start_acc = time.time()

def Accuracy():
    accNum = 0
    falsNum = 0
    for p in range(D.shape[0]):
        for e in range(D.shape[2]):
            z = D[p,:,e]
            pp, pe = TestImg(z)
            print(pp, p)
            if pp-p == 0:
                accNum = accNum+1
            else:
                falsNum = falsNum+1
    return accNum/(accNum+falsNum)

end_acc = time.time()
 
print("Tocnost: ",Accuracy())


# Dobar primjer1
#Pers = 9
#Expr = 6
# Dobar primjer2
#Pers = 5
#Expr = 9
Pers = randint(0,P-1)
Expr = randint(0,10-I-1)

def PlotImg(p, e):
    #ReconstructedImg1 = np.dot(np.dot(F, mode_dot(S,H[:,p],0)), G[:,e]) #rekonstruirana slika sa stupcima iz G
    #ReconstructedImg2 = np.dot(np.dot(F, mode_dot(S,H[p,:],0)), G[e,:]) #rekonstruirana slika sa retcima iz G
    #ReconstructedImg = mode_dot(mode_dot(mode_dot(S,F,1), G[:,e],2), H[:,p], 0)
    TestImg = D[Pers,:,Expr] #testna slika
    PredictImg = T[p,:,e] #najblizi match
    plt.figure()
    plt.imshow(TestImg.reshape(imgShape))
    plt.show()
    plt.figure()
    plt.imshow(PredictImg.reshape(imgShape))
    plt.show()
    # plt.plot(PredictImg.reshape(imgShape))
    #plt.figure()
    #plt.imshow(ReconstructedImg.reshape(imgShape))
    #plt.figure()
    #plt.imshow(ReconstructedImg2.reshape(imgShape))

z = D[Pers,:,Expr]

start_test = time.time() 
Predict_pers, Predict_expr = TestImg(z)
end_test = time.time() 

PlotImg(Predict_pers, Predict_expr) 

print("Vrijeme treniranja: ", str(end_train - start_train))
print("Vrijeme testiranja: ", str(end_test - start_test))
print("Vrijeme racunanja tocnosti: ", str(end_acc - start_acc))
